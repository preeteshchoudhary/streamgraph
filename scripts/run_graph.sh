#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python - <<'PY'
import platform,sys
if sys.version_info < (3,11):
    raise SystemExit('The pinned graph environment requires Python 3.11 or newer.')
if platform.system() == 'Darwin':
    version=[int(x) for x in platform.mac_ver()[0].split('.')]
    if platform.machine() != 'arm64' or tuple((version+[0,0])[:2]) < (14,0):
        raise SystemExit('The pinned PyTorch 2.12 Mac wheels require native Apple Silicon Python and macOS 14+. Use a supported Mac or Linux environment for this graph milestone; baseline commands remain available.')
PY
if [[ "$(uname -s)" == 'Linux' ]]; then
  python -m pip install 'torch==2.12.0' --index-url https://download.pytorch.org/whl/cpu
fi
python -m pip install -r requirements-graph.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
streamgraph doctor
if [[ ! -f data/processed/elliptic.npz ]]; then
  streamgraph download
  streamgraph prepare
fi
streamgraph train-graph
