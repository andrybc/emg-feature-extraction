"""
Artifact processors for the EMG simulator.

Each artifact models one physical phenomenon that contaminates surface
EMG. Artifacts share a common contract defined by BaseArtifact and are
chained in order inside the simulator's signal generation loop.

To add a new artifact:
    1. Subclass BaseArtifact in src/artifacts/<name>.py.
    2. Add it to the import list and __all__ below.
    3. Append an instance to self.artifacts in EMGSimulatorApp.__init__.

On main, only the BaseArtifact contract is exported. Concrete artifacts
live on their own branches (e.g. artifact/sweat) and add themselves
here when they merge.
"""

from src.artifacts.base import BaseArtifact

__all__ = ["BaseArtifact"]