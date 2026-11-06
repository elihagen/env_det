#!/usr/bin/env python3
"""Run a script under debugpy, waiting for VS Code to attach (the .vscode/launch.json configurations).

    python -Xfrozen_modules=off scripts/debug_listen.py scripts/train_salsanext.py --data ... --out ...
"""
import runpy
import sys
from pathlib import Path

import debugpy

PORT = 5678

debugpy.listen(('0.0.0.0', PORT))
print(f'debugpy waiting for VS Code on port {PORT}', flush=True)  # the VS Code task waits for this line
debugpy.wait_for_client()
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parent))  # scripts import their siblings
runpy.run_path(script, run_name='__main__')
