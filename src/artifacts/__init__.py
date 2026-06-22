"""
Artifact processors for the EMG simulator.

Each artifact models one physical phenomenon that contaminates surface
EMG. Artifacts share a common contract defined by BaseArtifact and are
chained in order inside the simulator's signal generation loop.

To add a new artifact:
    1. Subclass BaseArtifact in src/artifacts/<name>.py.
    2. Add it to the import list and __all__ below.
    3. Append an instance to self.artifacts in EMGSimulatorApp.__init__.

Exports on this branch:
    - BaseArtifact:   the contract (defined on main)
    - SweatProcessor: surface-EMG sweat artifact (this branch)
"""

from src.artifacts.base import BaseArtifact
from src.artifacts.sweat import SweatProcessor

__all__ = ["BaseArtifact", "SweatProcessor"]