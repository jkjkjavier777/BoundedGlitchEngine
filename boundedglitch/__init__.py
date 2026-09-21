"""BoundedGlitchEngine package."""
__version__ = "1.0.0"

from .config import load_config
from .conversation import ConversationManager

__all__ = ["load_config", "ConversationManager"]
from .engine import BoundedGlitchEngine
