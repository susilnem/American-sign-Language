import sys
from pathlib import Path

# app.py imports sibling modules (e.g. `from detector_worker import ...`) as
# bare names, relying on its own directory being on sys.path (true when run
# as `python src/app.py`). Tests import it as `src.app`, so add src/ here too.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
