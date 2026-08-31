"""Engine layer: pure Python, UI-agnostic.

Nothing in this package may import Qt. The UI is a thin client over these
modules, and ``tests/test_layering.py`` enforces the boundary by walking the
import graph of every module here.
"""
