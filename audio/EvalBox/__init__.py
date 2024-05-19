import sys
from pathlib import Path

for level in [1, 2]:
    if str(Path(__file__).resolve().parents[level]) not in sys.path:
        sys.path.append(str(Path(__file__).resolve().parents[level]))
del level
