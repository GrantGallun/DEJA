"""Game-agnostic HUD event discovery for Smart Replay.

Finds a game's event vocabulary (death screens, kill banners, round starts)
from recorded footage, with no per-game configuration, by exploiting the fact
that event HUD pixels are temporally distinguishable from world geometry:
they snap to a tight value and hold it for seconds at a time, repeatedly.
"""

__version__ = "0.2.6"
