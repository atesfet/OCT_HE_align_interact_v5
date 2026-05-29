from __future__ import annotations

"""Recovered entry point for the original align_he_to_oct_2D_v3 implementation.

The readable source for this script was accidentally deleted, but the compiled
Python 3.12 bytecode survived in ``src/__pycache__``. That bytecode was copied to
``src/recovered_bytecode`` before restoring this source filename so future Python
runs do not overwrite the only recovered copy.

This shim loads the recovered implementation and re-exports its public functions
and CLI. The registration behavior is therefore the original 2D_v3 behavior,
including rembg-based OCT/HE masks, optional manual masks, z-range arguments,
partial-boundary scoring, and the original output naming.
"""

import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RECOVERED_PYC = SCRIPT_DIR / "recovered_bytecode" / "align_he_to_oct_2D_v3.cpython-312.pyc"


def _load_recovered_module() -> ModuleType:
    if not RECOVERED_PYC.exists():
        raise FileNotFoundError(
            "Recovered align_he_to_oct_2D_v3 bytecode is missing: "
            f"{RECOVERED_PYC}"
        )
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    loader = importlib.machinery.SourcelessFileLoader("_recovered_align_he_to_oct_2D_v3", str(RECOVERED_PYC))
    return loader.load_module()


_impl = _load_recovered_module()

FEATURE_PRESET_NAME = _impl.FEATURE_PRESET_NAME
SearchResult = _impl.SearchResult

align_he_to_oct_2D_v3 = _impl.align_he_to_oct_2D_v3
_discover_2d_pairs = _impl._discover_2d_pairs
_append_status = _impl._append_status


def main() -> None:
    _impl.main()


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


if __name__ == "__main__":
    main()
