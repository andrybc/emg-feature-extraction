"""
Abstract base class for EMG artifact processors.

Every artifact in src/artifacts/ models one physical phenomenon that
contaminates surface EMG (sweat, motion, fatigue, electrode shift,
powerline interference, etc). Each one is a function from signal to
signal, optionally with internal state that must be cleared between
trials.

To add a new artifact, subclass BaseArtifact, implement process() and
reset(), and register it in src/artifacts/__init__.py.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseArtifact(ABC):
    """
    Contract for an EMG artifact processor.

    All artifacts must accept and return an array of shape
    (n_samples, n_channels) so they can be chained in any order
    inside the simulator's signal generation loop.
    """

    @abstractmethod
    def process(self, emg: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Apply this artifact's effect to a block of EMG.

        Parameters
        ----------
        emg : np.ndarray
            Input EMG, shape (n_samples, n_channels), units mV.
        sample_rate : float
            Sampling rate in Hz. Required because most artifacts depend
            on it (filter cutoffs, drift time constants, powerline phase).

        Returns
        -------
        np.ndarray
            Modified EMG, same shape and dtype as input.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Clear any internal state (filter memory, drift accumulators,
        random number generator, phase counters).

        Called at the start of a new recording or trial. Stateless
        artifacts should still implement this as a no-op rather than
        omitting it, to make the choice explicit.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"