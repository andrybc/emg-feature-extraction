"""
pytest configuration. Adds the repo root to sys.path so tests can import
the src package and the emg_simulator module without requiring
PYTHONPATH=. on every invocation.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
