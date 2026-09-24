"""Make `import cinescope` work when running `python -m pytest` from the repo root
without installing the package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
