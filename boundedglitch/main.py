import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boundedglitch.engine import BoundedGlitchEngine

if __name__ == "__main__":
    BoundedGlitchEngine().interactive_mode()
