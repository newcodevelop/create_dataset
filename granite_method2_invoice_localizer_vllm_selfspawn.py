#!/usr/bin/env python3
"""
granite_method2_invoice_localizer.py

Method-2 layout-preserving invoice localization/translation.

What this implements
--------------------
Given:
  1) a folder of line-OCR bbox files in SROIE format:
         x1,y1,x2,y2,x3,y3,x4,y4,text
  2) a folder of matching invoice images

The script:
  1. loads OCR line items x_i and bbox(x_i)
  2. asks Granite Vision 4.1 4B to extract semantic blocks/KV-like blocks
     from the original English invoice image, grounded to OCR line IDs
  3. translates/transliterates each semantic block as a whole with the SAME
     Granite model, asking the model to preserve the same number of lines
  4. renders the translated block line-by-line into the original OCR line bands
     on a blank white canvas

This is intentionally NOT the older line-level classifier pipeline.  The model
first creates block-level structure, then each block is translated as a unit,
while rendering preserves approximate line breaks using OCR line slots.

Example
-------
python granite_method2_invoice_localizer.py \
  --bbox-dir ./bbox \
  --image-dir ./images \
  --output-dir ./method2_hi \
  --target-language Hindi \
  --target-script Devanagari \
  --backend transformers \
  --device cuda \
  --font-path /usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf

Self-managed vLLM example
-------------------------
The script can spawn vLLM by itself, wait for the OpenAI-compatible endpoint,
run all inference, and shut the server down when the job finishes:

python granite_method2_invoice_localizer.py \
  --bbox-dir ./bbox \
  --image-dir ./images \
  --output-dir ./method2_hi \
  --target-language Hindi \
  --target-script Devanagari \
  --backend vllm \
  --device cuda \
  --font-path /usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf

You can still use an already-running vLLM/OpenAI-compatible server by passing
--backend openai and --openai-base-url.

Dependencies
------------
Core: pillow tqdm
Transformers backend: torch transformers>=5.8.0 peft>=0.19.1 tokenizers>=0.22.2 pillow>=12.2.0
Self-managed vLLM backend: vllm openai
OpenAI/vLLM server backend: openai
"""

from __future__ import annotations

import argparse
import atexit
import base64
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import warnings
from urllib.error import URLError
from urllib.request import urlopen
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class OCRLine:
    doc_id: str
    line_id: int
    bbox8: List[int]
    text: str
    image_w: int
    image_h: int

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        xs = self.bbox8[0::2]
        ys = self.bbox8[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def x0(self) -> int:
        return self.rect[0]

    @property
    def y0(self) -> int:
        return self.rect[1]

    @property
    def x1(self) -> int:
        return self.rect[2]

    @property
    def y1(self) -> int:
        return self.rect[3]

    @property
    def h(self) -> int:
        return max(1, self.y1 - self.y0)

    @property
    def w(self) -> int:
        return max(1, self.x1 - self.x0)


@dataclass
class SemanticBlock:
    block_id: str
    field: str
    action: str
    line_ids: List[int]
    source_lines: List[str]
    model_notes: str = ""
    raw_model_block: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderedLine:
    doc_id: str
    block_id: str
    field: str
    action: str
    line_id: int
    bbox8: List[int]
    slot_rect: List[int]
    original_text: str
    translated_text: str
    render_meta: Dict[str, Any]


# -----------------------------------------------------------------------------
# OCR and image I/O
# -----------------------------------------------------------------------------


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_sroie_ocr_file(path: Path, image_w: int, image_h: int) -> List[OCRLine]:
    """Parse x1,y1,x2,y2,x3,y3,x4,y4,text OCR files."""
    lines: List[OCRLine] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_idx, raw in enumerate(f):
            raw = raw.rstrip("\n\r")
            if not raw.strip():
                continue
            parts = raw.split(",", 8)
            if len(parts) < 9:
                warnings.warn(f"Skipping malformed OCR line {raw_idx + 1} in {path}: {raw[:120]}")
                continue
            try:
                coords = [int(round(float(v))) for v in parts[:8]]
            except ValueError:
                warnings.warn(f"Skipping OCR line with nonnumeric bbox {raw_idx + 1} in {path}: {raw[:120]}")
                continue
            text = parts[8].strip()
            if not text:
                continue
            lines.append(
                OCRLine(
                    doc_id=path.stem,
                    line_id=len(lines),
                    bbox8=coords,
                    text=text,
                    image_w=image_w,
                    image_h=image_h,
                )
            )
    return lines


def find_bbox_files(bbox_dir: Path) -> List[Path]:
    if bbox_dir.is_file():
        return [bbox_dir]
    files = sorted(p for p in bbox_dir.rglob("*.txt") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .txt bbox/OCR files found under {bbox_dir}")
    return files


def find_image_for_stem(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    for p in image_dir.rglob("*"):
        if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def load_doc(bbox_file: Path, image_dir: Path) -> Tuple[List[OCRLine], Image.Image, Path]:
    img_path = find_image_for_stem(image_dir, bbox_file.stem)
    if img_path is None:
        raise FileNotFoundError(f"Could not find matching image for {bbox_file.name} under {image_dir}")
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    lines = parse_sroie_ocr_file(bbox_file, w, h)
    return lines, image, img_path


# -----------------------------------------------------------------------------
# Granite model wrappers
# -----------------------------------------------------------------------------


class GraniteBackend:
    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError


class TransformersGraniteBackend(GraniteBackend):
    """Local HuggingFace Transformers backend for Granite Vision 4.1 4B."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ) -> None:
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        except Exception:
            # Older examples for this model used AutoModelForMultimodalLM.
            from transformers import AutoModelForMultimodalLM  # type: ignore
            model_cls = AutoModelForMultimodalLM

        self.torch = torch
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        torch_dtype = self._resolve_dtype(dtype)
        kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        # Transformers >= 5 uses dtype; many installed versions still use torch_dtype.
        kwargs_dtype_first = dict(kwargs)
        kwargs_dtype_first["dtype"] = torch_dtype
        kwargs_torch_dtype = dict(kwargs)
        kwargs_torch_dtype["torch_dtype"] = torch_dtype

        if device == "auto":
            kwargs_dtype_first["device_map"] = "auto"
            kwargs_torch_dtype["device_map"] = "auto"
        elif device.startswith("cuda"):
            kwargs_dtype_first["device_map"] = device
            kwargs_torch_dtype["device_map"] = device

        try:
            self.model = model_cls.from_pretrained(model_id, **kwargs_dtype_first).eval()
        except TypeError:
            self.model = model_cls.from_pretrained(model_id, **kwargs_torch_dtype).eval()

        if not (device == "auto" or device.startswith("cuda")):
            self.model.to(device)

    def _resolve_dtype(self, dtype: str):
        dtype = dtype.lower()
        if dtype in {"auto", "default"}:
            return "auto"
        if dtype in {"bf16", "bfloat16"}:
            return self.torch.bfloat16
        if dtype in {"fp16", "float16", "half"}:
            return self.torch.float16
        if dtype in {"fp32", "float32"}:
            return self.torch.float32
        raise ValueError(f"Unknown dtype: {dtype}")

    @property
    def device(self):
        try:
            return self.model.device
        except Exception:
            return next(self.model.parameters()).device

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            do_pad=True,
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        gen = outputs[0, inputs["input_ids"].shape[1] :]
        return self.processor.decode(gen, skip_special_tokens=True).strip()


class OpenAICompatibleGraniteBackend(GraniteBackend):
    """For vLLM/SGLang/OpenAI-compatible Granite Vision servers."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ) -> None:
        from openai import OpenAI

        self.model_id = model_id
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_new_tokens,
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()


class ManagedVLLMGraniteBackend(OpenAICompatibleGraniteBackend):
    """Start and stop a local vLLM server, then use the normal OpenAI client.

    The rest of the pipeline only needs a `generate(image, prompt, max_tokens)`
    method.  To keep the Method-2 logic unchanged, this class simply turns a
    self-launched vLLM process into the same OpenAI-compatible backend used by
    `OpenAICompatibleGraniteBackend`.

    Lifecycle:
      1. check whether a compatible endpoint is already alive at host:port;
      2. if not, spawn `vllm serve <model_id> ...`;
      3. poll `/health` and `/v1/models` until the endpoint is ready;
      4. send vision-language requests through the OpenAI Python client;
      5. terminate the vLLM process at the end unless `keep_alive=True`.
    """

    def __init__(
        self,
        model_id: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        startup_timeout: float = 900.0,
        serve_command: str = "vllm serve",
        dtype: str = "bfloat16",
        device: str = "cuda",
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None,
        served_model_name: Optional[str] = None,
        trust_remote_code: bool = True,
        extra_args: str = "",
        log_file: Optional[Path] = None,
        reuse_existing: bool = True,
        keep_alive: bool = False,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.base_url = f"http://{self.host}:{self.port}/v1"
        self.health_url = f"http://{self.host}:{self.port}/health"
        self.models_url = f"{self.base_url}/models"
        self.keep_alive = keep_alive
        self._process: Optional[subprocess.Popen] = None
        self._log_fh = None
        self._spawned_by_this_script = False

        # If the user gave a served-model-name, vLLM will expose that name through
        # the OpenAI endpoint.  The client must use that exposed name.
        client_model_name = served_model_name or model_id

        endpoint_already_ready = reuse_existing and self._endpoint_ready_quietly()
        if endpoint_already_ready:
            print(f"[vLLM] Reusing existing server at {self.base_url}")
        else:
            self._start_vllm_process(
                model_id=model_id,
                serve_command=serve_command,
                dtype=dtype,
                device=device,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                served_model_name=served_model_name,
                trust_remote_code=trust_remote_code,
                extra_args=extra_args,
                log_file=log_file,
            )
            self._wait_until_ready(startup_timeout=startup_timeout)

        # Once the endpoint exists, use the exact same OpenAI-compatible inference
        # path as a manually launched vLLM server.
        super().__init__(
            model_id=client_model_name,
            base_url=self.base_url,
            api_key=api_key,
            timeout=timeout,
        )
        atexit.register(self.close)

    def _start_vllm_process(
        self,
        model_id: str,
        serve_command: str,
        dtype: str,
        device: str,
        tensor_parallel_size: Optional[int],
        gpu_memory_utilization: Optional[float],
        max_model_len: Optional[int],
        served_model_name: Optional[str],
        trust_remote_code: bool,
        extra_args: str,
        log_file: Optional[Path],
    ) -> None:
        # Build `vllm serve <model> ...` as a list rather than a shell string.
        # This avoids shell quoting bugs and makes the command safer.
        cmd = shlex.split(serve_command) + [model_id, "--host", self.host, "--port", str(self.port)]

        # vLLM and Transformers dtype names differ slightly.  vLLM accepts
        # bfloat16/float16/float32/auto.
        if dtype and dtype not in {"default", "none"}:
            dtype_for_vllm = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(dtype, dtype)
            cmd += ["--dtype", dtype_for_vllm]

        if tensor_parallel_size is not None:
            cmd += ["--tensor-parallel-size", str(tensor_parallel_size)]
        if gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
        if max_model_len is not None:
            cmd += ["--max-model-len", str(max_model_len)]
        if served_model_name:
            cmd += ["--served-model-name", served_model_name]
        if trust_remote_code:
            cmd += ["--trust-remote-code"]
        if extra_args.strip():
            cmd += shlex.split(extra_args)

        # Device selection: if the caller passed --device cuda:1, restrict the
        # spawned vLLM process to that GPU via CUDA_VISIBLE_DEVICES.  For normal
        # --device cuda, leave the environment untouched.
        env = os.environ.copy()
        m = re.fullmatch(r"cuda:(\d+)", device.strip())
        if m:
            env["CUDA_VISIBLE_DEVICES"] = m.group(1)

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = log_file.open("a", encoding="utf-8")
            stdout = self._log_fh
            stderr = subprocess.STDOUT
            print(f"[vLLM] Logs: {log_file}")
        else:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL

        print("[vLLM] Starting server:", " ".join(shlex.quote(x) for x in cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=stderr,
            env=env,
            start_new_session=True,
        )
        self._spawned_by_this_script = True

    def _endpoint_ready_quietly(self) -> bool:
        try:
            self._open_url(self.health_url, timeout=2.0)
            return True
        except Exception:
            try:
                self._open_url(self.models_url, timeout=2.0)
                return True
            except Exception:
                return False

    @staticmethod
    def _open_url(url: str, timeout: float) -> bytes:
        with urlopen(url, timeout=timeout) as r:
            return r.read()

    def _wait_until_ready(self, startup_timeout: float) -> None:
        print(f"[vLLM] Waiting for endpoint at {self.base_url} ...")
        start = time.time()
        last_err = ""
        while time.time() - start < startup_timeout:
            # If vLLM dies during startup, fail early and show the log tail.
            if self._process is not None and self._process.poll() is not None:
                tail = self._tail_log()
                raise RuntimeError(
                    f"vLLM process exited with code {self._process.returncode} before becoming ready.\n"
                    f"Last error: {last_err}\n"
                    f"Log tail:\n{tail}"
                )
            try:
                self._open_url(self.health_url, timeout=5.0)
                print("[vLLM] Server is ready.")
                return
            except Exception as e:
                last_err = repr(e)
                # Some vLLM builds expose /v1/models earlier/more reliably than
                # /health, so accept either.
                try:
                    self._open_url(self.models_url, timeout=5.0)
                    print("[vLLM] Server is ready.")
                    return
                except Exception as e2:
                    last_err = repr(e2)
            time.sleep(3.0)

        tail = self._tail_log()
        raise TimeoutError(
            f"Timed out after {startup_timeout:.0f}s waiting for vLLM at {self.base_url}.\n"
            f"Last error: {last_err}\n"
            f"Log tail:\n{tail}"
        )

    def _tail_log(self, n_chars: int = 4000) -> str:
        if self._log_fh is None:
            return "<no log file configured; pass --vllm-log-file to capture startup logs>"
        try:
            self._log_fh.flush()
            path = Path(self._log_fh.name)
            data = path.read_text(encoding="utf-8", errors="ignore")
            return data[-n_chars:]
        except Exception as e:
            return f"<could not read log: {e}>"

    def close(self) -> None:
        # Do not kill a server that existed before this script started.  Only clean
        # up the process that we spawned ourselves.
        if self.keep_alive or not self._spawned_by_this_script or self._process is None:
            return
        if self._process.poll() is not None:
            return
        print("[vLLM] Stopping server spawned by this script ...")
        try:
            os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
        except Exception:
            self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            except Exception:
                self._process.kill()
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Prompting and JSON parsing
# -----------------------------------------------------------------------------


def ocr_lines_for_prompt(lines: Sequence[OCRLine]) -> str:
    rows = []
    for l in lines:
        x0, y0, x1, y1 = l.rect
        rows.append(f"[{l.line_id}] bbox=({x0},{y0},{x1},{y1}) text={json.dumps(l.text, ensure_ascii=False)}")
    return "\n".join(rows)


def build_extraction_prompt(lines: Sequence[OCRLine]) -> str:
    return f"""
You are extracting semantic blocks from an English invoice/receipt for layout-preserving translation.
Use the image and the OCR line list. The OCR line IDs are authoritative.

Return ONLY valid JSON, no markdown, no explanation.

JSON format:
{{
  "blocks": [
    {{
      "block_id": "B0",
      "field": "vendor_name | vendor_address | invoice_title | invoice_number | date_time | table_header | table_row | tax_summary | total | footer_terms | other",
      "action": "translate | transliterate | copy | mixed",
      "line_ids": [0, 1],
      "source_lines": ["exact OCR text for line 0", "exact OCR text for line 1"],
      "notes": "short reason"
    }}
  ]
}}

Rules:
1. Group consecutive OCR lines into semantic blocks/KV-like blocks.
2. Address blocks should contain all address lines together.
3. Table headers may be one block. Table rows should usually be separate row-level blocks, not one giant table block.
4. Pure numbers, dates, invoice IDs, phone numbers, emails, tax IDs, amounts, quantities, and currency symbols should have action="copy".
5. Names, vendor/customer names, addresses, and product names should usually have action="transliterate".
6. Generic labels and sentences such as "Tax Invoice", "Description", "Total", "Thank you" should have action="translate".
7. Lines that combine labels/text with numbers or table cells should have action="mixed".
8. Every OCR line ID must appear in exactly one block. Do not invent text.
9. Preserve original reading order.

OCR lines:
{ocr_lines_for_prompt(lines)}
""".strip()


def build_translation_prompt(
    block: SemanticBlock,
    target_language: str,
    target_script: str,
    strict_line_count: bool = True,
) -> str:
    numbered_lines = "\n".join(
        f"[{i}] {text}" for i, text in enumerate(block.source_lines)
    )
    k = len(block.source_lines)
    same_line_rule = (
        f"Return exactly {k} translated lines in the same order. The output list length must be {k}."
        if strict_line_count
        else "Preserve line breaks as much as possible."
    )
    return f"""
You are localizing one semantic block from an English invoice/receipt into {target_language} using {target_script} script.
The block action is: {block.action}.
The semantic field is: {block.field}.

Return ONLY valid JSON, no markdown, no explanation:
{{"lines": ["line 0 output", "line 1 output"]}}

Rules:
1. {same_line_rule}
2. Preserve line identity, order, and approximate line-wise content.
3. Copy numbers, dates, times, currency amounts, invoice IDs, phone numbers, emails, URLs, GST/VAT/TIN/PAN-like IDs, product codes, and tax codes exactly.
4. If action="translate", translate meaning into {target_language}.
5. If action="transliterate", phonetically render names, company names, product names, and addresses in {target_script}; still copy numbers/codes exactly.
6. If action="mixed", translate labels/common words, transliterate names/product/address text, and copy numbers/codes exactly.
7. If action="copy", return the original lines unchanged.
8. Do not add explanations, prefixes, bullets, or extra lines.

Source lines:
{numbered_lines}
""".strip()


def strip_code_fences(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def extract_json_text(text: str) -> str:
    text = strip_code_fences(text)
    start_candidates = [i for i in [text.find("{"), text.find("[")] if i >= 0]
    if not start_candidates:
        return text
    start = min(start_candidates)
    # Pick matching final brace/bracket. This intentionally tolerates stray prefix/suffix.
    if text[start] == "{":
        end = text.rfind("}")
    else:
        end = text.rfind("]")
    if end > start:
        return text[start : end + 1]
    return text[start:]


def load_json_from_model(text: str) -> Any:
    js = extract_json_text(text)
    try:
        return json.loads(js)
    except json.JSONDecodeError as e:
        # common repair: trailing commas
        repaired = re.sub(r",\s*([}\]])", r"\1", js)
        try:
            return json.loads(repaired)
        except Exception:
            raise ValueError(f"Could not parse model JSON. Error: {e}. Raw output begins:\n{text[:1000]}")


def normalize_action(action: Any) -> str:
    a = str(action or "mixed").strip().lower()
    aliases = {
        "translation": "translate",
        "translated": "translate",
        "transliteration": "transliterate",
        "transliterated": "transliterate",
        "unchanged": "copy",
        "keep": "copy",
    }
    a = aliases.get(a, a)
    if a not in {"translate", "transliterate", "copy", "mixed"}:
        return "mixed"
    return a


# -----------------------------------------------------------------------------
# Alignment and block validation
# -----------------------------------------------------------------------------


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def fuzzy_ratio(a: str, b: str) -> float:
    na, nb = norm_text(a), norm_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return min(len(na), len(nb)) / max(len(na), len(nb))
    return SequenceMatcher(None, na, nb).ratio()


def match_source_lines_to_ids(source_lines: Sequence[str], ocr_lines: Sequence[OCRLine], used: set[int]) -> List[int]:
    ids: List[int] = []
    for src in source_lines:
        best_id = None
        best_score = -1.0
        for l in ocr_lines:
            if l.line_id in used or l.line_id in ids:
                continue
            score = fuzzy_ratio(src, l.text)
            if score > best_score:
                best_score = score
                best_id = l.line_id
        if best_id is not None and best_score >= 0.55:
            ids.append(best_id)
    return ids


def normalize_blocks(model_obj: Any, ocr_lines: Sequence[OCRLine]) -> List[SemanticBlock]:
    if isinstance(model_obj, dict):
        raw_blocks = model_obj.get("blocks", [])
    elif isinstance(model_obj, list):
        raw_blocks = model_obj
    else:
        raw_blocks = []

    by_id = {l.line_id: l for l in ocr_lines}
    used: set[int] = set()
    blocks: List[SemanticBlock] = []

    for idx, rb in enumerate(raw_blocks):
        if not isinstance(rb, dict):
            continue
        raw_ids = rb.get("line_ids", rb.get("lines", rb.get("line_id", [])))
        if isinstance(raw_ids, int):
            raw_ids = [raw_ids]
        if not isinstance(raw_ids, list):
            raw_ids = []
        ids = []
        for v in raw_ids:
            try:
                iv = int(v)
            except Exception:
                continue
            if iv in by_id and iv not in ids and iv not in used:
                ids.append(iv)

        raw_source = rb.get("source_lines", rb.get("text_lines", rb.get("source_text", [])))
        if isinstance(raw_source, str):
            source_lines = [s for s in raw_source.split("\n") if s.strip()]
        elif isinstance(raw_source, list):
            source_lines = [str(s).strip() for s in raw_source if str(s).strip()]
        else:
            source_lines = []

        if not ids and source_lines:
            ids = match_source_lines_to_ids(source_lines, ocr_lines, used)

        if not ids:
            continue

        ids = sorted(ids, key=lambda i: (by_id[i].y0, by_id[i].x0))
        for i in ids:
            used.add(i)
        if not source_lines or len(source_lines) != len(ids):
            source_lines = [by_id[i].text for i in ids]

        blocks.append(
            SemanticBlock(
                block_id=str(rb.get("block_id") or f"B{len(blocks)}"),
                field=str(rb.get("field") or rb.get("key") or "other"),
                action=normalize_action(rb.get("action")),
                line_ids=ids,
                source_lines=source_lines,
                model_notes=str(rb.get("notes") or rb.get("reason") or ""),
                raw_model_block=rb,
            )
        )

    # Guarantee coverage. This is a safety net for invalid model JSON; it does not
    # change the core method: these are still semantic blocks, just singletons.
    for l in ocr_lines:
        if l.line_id not in used:
            blocks.append(
                SemanticBlock(
                    block_id=f"B_missing_{l.line_id}",
                    field="other",
                    action="mixed",
                    line_ids=[l.line_id],
                    source_lines=[l.text],
                    model_notes="added_by_validator_missing_from_model_output",
                    raw_model_block={},
                )
            )

    blocks.sort(key=lambda b: (min(by_id[i].y0 for i in b.line_ids), min(by_id[i].x0 for i in b.line_ids)))
    # Reassign clean block IDs in reading order.
    for i, b in enumerate(blocks):
        b.block_id = f"B{i:03d}"
    return blocks


def heuristic_blocks_from_geometry(ocr_lines: Sequence[OCRLine]) -> List[SemanticBlock]:
    """Fallback only when model extraction JSON fails.

    Groups nearby lines into paragraph-like blocks. This keeps the method block-level
    rather than reverting to the older per-line classifier.
    """
    if not ocr_lines:
        return []
    ordered = sorted(ocr_lines, key=lambda l: (l.y0, l.x0))
    median_h = statistics.median([l.h for l in ordered])
    blocks: List[List[OCRLine]] = []
    cur: List[OCRLine] = [ordered[0]]
    for prev, line in zip(ordered, ordered[1:]):
        gap = line.y0 - prev.y1
        x_overlap = max(0, min(prev.x1, line.x1) - max(prev.x0, line.x0))
        overlap_frac = x_overlap / max(1, min(prev.w, line.w))
        same_paragraph = gap <= 1.25 * median_h and (overlap_frac > 0.15 or abs(line.x0 - prev.x0) < 0.20 * max(line.image_w, 1))
        if same_paragraph:
            cur.append(line)
        else:
            blocks.append(cur)
            cur = [line]
    blocks.append(cur)

    out: List[SemanticBlock] = []
    for i, group in enumerate(blocks):
        out.append(
            SemanticBlock(
                block_id=f"H{i:03d}",
                field="geometry_block",
                action="mixed",
                line_ids=[l.line_id for l in group],
                source_lines=[l.text for l in group],
                model_notes="heuristic fallback because model extraction failed",
            )
        )
    return out


# -----------------------------------------------------------------------------
# Translation line-count repair and wrapping
# -----------------------------------------------------------------------------


def parse_translation_output(raw: str, expected_k: int) -> List[str]:
    try:
        obj = load_json_from_model(raw)
        if isinstance(obj, dict):
            val = obj.get("lines", obj.get("translated_lines", obj.get("output_lines")))
        else:
            val = obj
        if isinstance(val, list):
            return [str(x).strip() for x in val]
        if isinstance(val, str):
            return [s.strip() for s in val.split("\n") if s.strip()]
    except Exception:
        pass
    # Fallback: split raw response by newlines, removing common bullets.
    lines = []
    for s in strip_code_fences(raw).splitlines():
        s = re.sub(r"^\s*(?:[-*•]|\d+[.)]|\[\d+\])\s*", "", s).strip()
        if s and not s.startswith("{") and not s.endswith("}"):
            lines.append(s)
    return lines[:expected_k] if lines else []


def repair_line_count(translated_lines: List[str], source_lines: List[str], k: int) -> List[str]:
    translated_lines = [s.strip() for s in translated_lines if str(s).strip()]
    if len(translated_lines) == k:
        return translated_lines
    if not translated_lines:
        return list(source_lines)
    merged = " ".join(translated_lines)
    return split_text_into_k_roughly_equal_lines(merged, k)


def split_text_into_k_roughly_equal_lines(text: str, k: int) -> List[str]:
    words = text.split()
    if k <= 1:
        return [text.strip()]
    if not words:
        return [""] * k
    # Greedy balance by character length.
    total = sum(len(w) for w in words) + max(0, len(words) - 1)
    target = max(1, math.ceil(total / k))
    out: List[str] = []
    cur: List[str] = []
    cur_len = 0
    remaining_slots = k
    for idx, w in enumerate(words):
        remaining_words = len(words) - idx
        cand_len = cur_len + (1 if cur else 0) + len(w)
        must_leave_words = remaining_words <= remaining_slots
        if cur and cand_len > target and not must_leave_words and len(out) < k - 1:
            out.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
            remaining_slots = k - len(out)
        else:
            cur.append(w)
            cur_len = cand_len
    if cur:
        out.append(" ".join(cur))
    while len(out) < k:
        out.append("")
    if len(out) > k:
        out = out[: k - 1] + [" ".join(out[k - 1 :])]
    return out


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansGurmukhi-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
    "/System/Library/Fonts/Devanagari Sangam MN.ttc",
    "/Library/Fonts/NotoSansDevanagari-Regular.ttf",
]


def resolve_font_path(font_path: Optional[str]) -> str:
    if font_path:
        p = Path(font_path)
        if not p.exists():
            raise FileNotFoundError(f"Font path does not exist: {font_path}")
        return str(p)
    for cand in FONT_CANDIDATES:
        if Path(cand).exists():
            return cand
    raise FileNotFoundError("No suitable font found automatically. Pass --font-path.")


def text_wh(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    if not text:
        return 0, 0
    box = draw.textbbox((0, 0), text, font=font)
    return int(box[2] - box[0]), int(box[3] - box[1])


def wrap_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> str:
    words = text.split()
    if len(words) <= 1:
        return text
    lines: List[str] = []
    cur = ""
    for w in words:
        cand = w if not cur else f"{cur} {w}"
        tw, _ = text_wh(draw, cand, font)
        if tw <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def draw_fitted_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    rect: Tuple[int, int, int, int],
    font_path: str,
    min_font_size: int,
    font_scale: float,
    allow_internal_wrap: bool = False,
) -> Dict[str, Any]:
    x0, y0, x1, y1 = rect
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    start_size = max(min_font_size, int(round(box_h * font_scale)))

    for size in range(start_size, min_font_size - 1, -1):
        font = ImageFont.truetype(font_path, size=size)
        candidate = text
        tw, th = text_wh(draw, candidate, font)
        if tw <= box_w and th <= box_h:
            y = y0 + max(0, (box_h - th) // 2)
            draw.text((x0, y), candidate, font=font, fill=0)
            return {"font_size": size, "fit_policy": "shrink", "overflow": False, "rendered_rect": [x0, y, x0 + tw, y + th]}

    if allow_internal_wrap:
        for size in range(start_size, min_font_size - 1, -1):
            font = ImageFont.truetype(font_path, size=size)
            candidate = wrap_to_width(draw, text, font, box_w)
            widths_heights = [text_wh(draw, ln, font) for ln in candidate.split("\n")]
            tw = max((w for w, _ in widths_heights), default=0)
            # Conservative line height.
            line_h = max((h for _, h in widths_heights), default=size)
            th = line_h * len(widths_heights)
            if tw <= box_w and th <= box_h:
                y = y0 + max(0, (box_h - th) // 2)
                for ln in candidate.split("\n"):
                    draw.text((x0, y), ln, font=font, fill=0)
                    y += line_h
                return {"font_size": size, "fit_policy": "internal_wrap", "overflow": False, "rendered_rect": [x0, y0, x0 + tw, y0 + th]}

    size = min_font_size
    font = ImageFont.truetype(font_path, size=size)
    tw, th = text_wh(draw, text, font)
    draw.text((x0, y0), text, font=font, fill=0)
    return {"font_size": size, "fit_policy": "overflow_min_font", "overflow": True, "rendered_rect": [x0, y0, x0 + tw, y0 + th]}


def union_rect(lines: Sequence[OCRLine]) -> Tuple[int, int, int, int]:
    return (
        min(l.x0 for l in lines),
        min(l.y0 for l in lines),
        max(l.x1 for l in lines),
        max(l.y1 for l in lines),
    )


def line_slot_for_block(line: OCRLine, block_lines: Sequence[OCRLine], pad_x: int = 1, pad_y: int = 1) -> Tuple[int, int, int, int]:
    ux0, _, ux1, _ = union_rect(block_lines)
    # Method 2: preserve the original vertical line band, but allow the line to
    # use the block's full horizontal support. This is more robust than forcing
    # translation into the exact narrow OCR line bbox.
    return (
        max(0, ux0 - pad_x),
        max(0, line.y0 - pad_y),
        min(line.image_w, ux1 + pad_x),
        min(line.image_h, line.y1 + pad_y),
    )


def render_method2_document(
    ocr_lines: Sequence[OCRLine],
    blocks: Sequence[SemanticBlock],
    translated_by_block: Dict[str, List[str]],
    image_size: Tuple[int, int],
    font_path: str,
    min_font_size: int,
    font_scale: float,
) -> Tuple[Image.Image, List[RenderedLine]]:
    by_id = {l.line_id: l for l in ocr_lines}
    canvas = Image.new("L", image_size, color=255)
    draw = ImageDraw.Draw(canvas)
    rendered: List[RenderedLine] = []

    for b in blocks:
        block_lines = [by_id[i] for i in b.line_ids if i in by_id]
        if not block_lines:
            continue
        outputs = translated_by_block.get(b.block_id, b.source_lines)
        outputs = repair_line_count(outputs, b.source_lines, len(block_lines))
        for line, out_text in zip(block_lines, outputs):
            slot = line_slot_for_block(line, block_lines)
            meta = draw_fitted_text(
                draw,
                out_text,
                slot,
                font_path=font_path,
                min_font_size=min_font_size,
                font_scale=font_scale,
                allow_internal_wrap=False,
            )
            rendered.append(
                RenderedLine(
                    doc_id=line.doc_id,
                    block_id=b.block_id,
                    field=b.field,
                    action=b.action,
                    line_id=line.line_id,
                    bbox8=line.bbox8,
                    slot_rect=list(slot),
                    original_text=line.text,
                    translated_text=out_text,
                    render_meta=meta,
                )
            )
    rendered.sort(key=lambda r: r.line_id)
    return canvas, rendered


# -----------------------------------------------------------------------------
# Save helpers
# -----------------------------------------------------------------------------


def save_sroie_txt(rendered: Sequence[RenderedLine], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in sorted(rendered, key=lambda x: x.line_id):
            coords = ",".join(str(int(v)) for v in r.bbox8)
            f.write(f"{coords},{r.translated_text}\n")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def block_to_dict(b: SemanticBlock) -> Dict[str, Any]:
    return asdict(b)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def translate_block_with_retries(
    backend: GraniteBackend,
    image: Image.Image,
    block: SemanticBlock,
    target_language: str,
    target_script: str,
    max_new_tokens: int,
    retries: int = 1,
    sleep_s: float = 0.5,
) -> Tuple[List[str], str]:
    last_raw = ""
    for attempt in range(retries + 1):
        prompt = build_translation_prompt(block, target_language, target_script, strict_line_count=True)
        raw = backend.generate(image, prompt, max_new_tokens=max_new_tokens)
        last_raw = raw
        lines = parse_translation_output(raw, expected_k=len(block.source_lines))
        if len(lines) == len(block.source_lines):
            return lines, raw
        time.sleep(sleep_s)
    return repair_line_count(parse_translation_output(last_raw, len(block.source_lines)), block.source_lines, len(block.source_lines)), last_raw


def process_one_document(
    bbox_file: Path,
    image_dir: Path,
    out_dir: Path,
    backend: GraniteBackend,
    target_language: str,
    target_script: str,
    font_path: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    ocr_lines, image, img_path = load_doc(bbox_file, image_dir)
    if not ocr_lines:
        raise ValueError(f"No OCR lines parsed from {bbox_file}")

    # Stage 1: ask Granite to create semantic blocks and map each block back to
    # authoritative OCR line IDs.  This is the key difference from the older
    # line-wise decision pipeline.
    extraction_prompt = build_extraction_prompt(ocr_lines)
    raw_extraction = backend.generate(image, extraction_prompt, max_new_tokens=args.max_extraction_tokens)

    extraction_failed = False
    try:
        extraction_obj = load_json_from_model(raw_extraction)
        blocks = normalize_blocks(extraction_obj, ocr_lines)
    except Exception as e:
        extraction_failed = True
        warnings.warn(f"Model extraction JSON failed for {bbox_file.name}: {e}")
        if not args.allow_heuristic_fallback:
            raise
        blocks = heuristic_blocks_from_geometry(ocr_lines)
        extraction_obj = {"blocks": [block_to_dict(b) for b in blocks], "fallback_reason": str(e)}

    # Stage 2: translate/transliterate each semantic block as a whole, but ask
    # Granite to return exactly one output line per source OCR line.  That gives
    # us semantic context during translation and line-level anchors for rendering.
    translated_by_block: Dict[str, List[str]] = {}
    raw_translation_by_block: Dict[str, str] = {}
    for block in blocks:
        if block.action == "copy":
            translated = list(block.source_lines)
            raw = json.dumps({"lines": translated}, ensure_ascii=False)
        else:
            translated, raw = translate_block_with_retries(
                backend=backend,
                image=image,
                block=block,
                target_language=target_language,
                target_script=target_script,
                max_new_tokens=args.max_translation_tokens,
                retries=args.translation_retries,
            )
        translated_by_block[block.block_id] = repair_line_count(translated, block.source_lines, len(block.source_lines))
        raw_translation_by_block[block.block_id] = raw

    # Stage 3: render on a blank white canvas.  Each translated line goes into
    # the original line's vertical band but can use the whole semantic block's
    # horizontal support.
    rendered_img, rendered_lines = render_method2_document(
        ocr_lines=ocr_lines,
        blocks=blocks,
        translated_by_block=translated_by_block,
        image_size=image.size,
        font_path=font_path,
        min_font_size=args.min_font_size,
        font_scale=args.font_scale,
    )

    out_img = out_dir / "images" / f"{bbox_file.stem}.png"
    out_txt = out_dir / "txt" / f"{bbox_file.stem}.txt"
    out_json = out_dir / "json" / f"{bbox_file.stem}.json"
    out_blocks = out_dir / "blocks" / f"{bbox_file.stem}.blocks.json"

    out_img.parent.mkdir(parents=True, exist_ok=True)
    rendered_img.save(out_img)
    save_sroie_txt(rendered_lines, out_txt)

    doc_json = {
        "doc_id": bbox_file.stem,
        "source_bbox_file": str(bbox_file),
        "source_image_file": str(img_path),
        "target_language": target_language,
        "target_script": target_script,
        "method": "method_2_block_level_translation_line_slot_rendering",
        "extraction_failed_and_used_fallback": extraction_failed,
        "ocr_lines": [asdict(l) for l in ocr_lines],
        "blocks": [block_to_dict(b) for b in blocks],
        "translated_by_block": translated_by_block,
        "rendered_lines": [asdict(r) for r in rendered_lines],
        "outputs": {
            "image": str(out_img),
            "txt": str(out_txt),
            "json": str(out_json),
            "blocks": str(out_blocks),
        },
    }
    save_json(doc_json, out_json)
    save_json(
        {
            "raw_extraction": raw_extraction,
            "parsed_extraction": extraction_obj,
            "normalized_blocks": [block_to_dict(b) for b in blocks],
            "raw_translation_by_block": raw_translation_by_block,
        },
        out_blocks,
    )

    n_overflow = sum(1 for r in rendered_lines if r.render_meta.get("overflow"))
    return {
        "doc_id": bbox_file.stem,
        "num_ocr_lines": len(ocr_lines),
        "num_blocks": len(blocks),
        "num_rendered_lines": len(rendered_lines),
        "num_overflow_lines": n_overflow,
        "overflow_rate": n_overflow / max(1, len(rendered_lines)),
        "source_image_file": str(img_path),
        "output_image_file": str(out_img),
        "output_json_file": str(out_json),
        "extraction_failed_and_used_fallback": extraction_failed,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Method-2 Granite Vision invoice localizer: block-level translation + line-slot rendering.")
    ap.add_argument("--bbox-dir", type=str, required=True, help="Folder/file containing SROIE-style OCR bbox txt files.")
    ap.add_argument("--image-dir", type=str, required=True, help="Folder containing matching invoice images.")
    ap.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--model-id", type=str, default="ibm-granite/granite-vision-4.1-4b")
    ap.add_argument(
        "--backend",
        choices=["vllm", "transformers", "openai"],
        default="vllm",
        help=(
            "vllm = spawn a local vLLM server and call it through OpenAI-compatible API; "
            "openai = use an already running OpenAI-compatible server; "
            "transformers = load the model directly in this Python process."
        ),
    )
    ap.add_argument("--device", type=str, default="cuda", help="cuda, cuda:0, cpu, or auto for Transformers backend; cuda:N also sets CUDA_VISIBLE_DEVICES for managed vLLM.")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    ap.add_argument("--openai-base-url", type=str, default="http://localhost:8000/v1", help="Endpoint for --backend openai only.")
    ap.add_argument("--openai-api-key", type=str, default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--openai-timeout", type=float, default=120.0)

    # Self-managed vLLM server options.  These only matter for --backend vllm.
    # They are exposed so you can tune GPU placement/memory/model length without
    # editing the script.
    ap.add_argument("--vllm-host", type=str, default="127.0.0.1")
    ap.add_argument("--vllm-port", type=int, default=8000)
    ap.add_argument("--vllm-serve-command", type=str, default="vllm serve", help="Command prefix used to launch vLLM. Default: 'vllm serve'.")
    ap.add_argument("--vllm-startup-timeout", type=float, default=900.0, help="Seconds to wait for vLLM model loading.")
    ap.add_argument("--vllm-log-file", type=str, default=None, help="Where to write vLLM server logs. Default: <output-dir>/vllm_server.log")
    ap.add_argument("--vllm-tensor-parallel-size", type=int, default=None)
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--vllm-max-model-len", type=int, default=None)
    ap.add_argument("--vllm-served-model-name", type=str, default=None, help="Optional alias exposed by vLLM and used by the OpenAI client.")
    ap.add_argument("--vllm-extra-args", type=str, default="", help="Extra arguments appended to the vLLM serve command, e.g. '--max-num-seqs 1'.")
    ap.add_argument("--vllm-reuse-existing", action="store_true", default=True, help="Reuse an already-ready server on host:port instead of spawning another one.")
    ap.add_argument("--no-vllm-reuse-existing", dest="vllm_reuse_existing", action="store_false")
    ap.add_argument("--keep-vllm-alive", action="store_true", help="Do not terminate the vLLM server after processing.")
    ap.add_argument("--no-vllm-trust-remote-code", dest="vllm_trust_remote_code", action="store_false", default=True)

    ap.add_argument("--target-language", type=str, default="Hindi")
    ap.add_argument("--target-script", type=str, default="Devanagari")
    ap.add_argument("--font-path", type=str, default=None)
    ap.add_argument("--min-font-size", type=int, default=6)
    ap.add_argument("--font-scale", type=float, default=0.92, help="Initial font size as a fraction of OCR line-band height.")

    ap.add_argument("--max-extraction-tokens", type=int, default=4096)
    ap.add_argument("--max-translation-tokens", type=int, default=512)
    ap.add_argument("--translation-retries", type=int, default=1)
    ap.add_argument("--allow-heuristic-fallback", action="store_true", default=True)
    ap.add_argument("--no-heuristic-fallback", dest="allow_heuristic_fallback", action="store_false")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    bbox_dir = Path(args.bbox_dir)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    font_path = resolve_font_path(args.font_path)
    print(f"Using font: {font_path}")
    print(f"Target: {args.target_language} / {args.target_script}")
    print(f"Model: {args.model_id} via {args.backend}")

    # Build one backend object and reuse it for all documents.
    # This is important for vLLM because model loading is expensive; we want one
    # server for the whole folder, not one server per invoice.
    backend: GraniteBackend
    if args.backend == "transformers":
        backend = TransformersGraniteBackend(
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=True,
        )
    elif args.backend == "openai":
        backend = OpenAICompatibleGraniteBackend(
            model_id=args.model_id,
            base_url=args.openai_base_url,
            api_key=args.openai_api_key,
            timeout=args.openai_timeout,
        )
    else:
        vllm_log_file = Path(args.vllm_log_file) if args.vllm_log_file else out_dir / "vllm_server.log"
        backend = ManagedVLLMGraniteBackend(
            model_id=args.model_id,
            host=args.vllm_host,
            port=args.vllm_port,
            api_key=args.openai_api_key,
            timeout=args.openai_timeout,
            startup_timeout=args.vllm_startup_timeout,
            serve_command=args.vllm_serve_command,
            dtype=args.dtype,
            device=args.device,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            served_model_name=args.vllm_served_model_name,
            trust_remote_code=args.vllm_trust_remote_code,
            extra_args=args.vllm_extra_args,
            log_file=vllm_log_file,
            reuse_existing=args.vllm_reuse_existing,
            keep_alive=args.keep_vllm_alive,
        )

    files = find_bbox_files(bbox_dir)
    if args.limit is not None:
        files = files[: args.limit]

    summaries: List[Dict[str, Any]] = []
    for bbox_file in tqdm(files, desc="Method-2 localizing invoices"):
        try:
            summaries.append(
                process_one_document(
                    bbox_file=bbox_file,
                    image_dir=image_dir,
                    out_dir=out_dir,
                    backend=backend,
                    target_language=args.target_language,
                    target_script=args.target_script,
                    font_path=font_path,
                    args=args,
                )
            )
        except Exception as e:
            warnings.warn(f"FAILED {bbox_file}: {e}")
            summaries.append({"doc_id": bbox_file.stem, "error": str(e)})

    summary = {
        "method": "method_2_block_level_translation_line_slot_rendering",
        "model_id": args.model_id,
        "backend": args.backend,
        "vllm_base_url": f"http://{args.vllm_host}:{args.vllm_port}/v1" if args.backend == "vllm" else None,
        "target_language": args.target_language,
        "target_script": args.target_script,
        "bbox_dir": str(bbox_dir),
        "image_dir": str(image_dir),
        "output_dir": str(out_dir),
        "font_path": font_path,
        "num_documents": len(files),
        "documents": summaries,
    }
    save_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Explicitly shut down self-managed vLLM here.  The atexit hook is only a
    # safety net; explicit close makes the lifecycle clear and avoids orphaned
    # GPU processes in notebook/cluster jobs.
    if hasattr(backend, "close"):
        try:
            backend.close()  # type: ignore[attr-defined]
        except Exception as e:
            warnings.warn(f"Could not close backend cleanly: {e}")


if __name__ == "__main__":
    main()
