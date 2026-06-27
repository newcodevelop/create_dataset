#!/usr/bin/env bash
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python - <<'PY'
from ai4bharat.transliteration import XlitEngine

e = XlitEngine(lang2use="hi", beam_width=10, rescore=False)
print("engine ok")

for w in ["tan", "woon", "yann", "selangor", "balakong", "kawasan", "perindustrian"]:
    try:
        print(w, "=>", e.translit_word(w, topk=5))
    except TypeError:
        print(w, "=>", e.translit_word(w))
PY
