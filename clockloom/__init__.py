"""Fit a recorder clock and render it on a common timeline."""

from .alignment import AlignmentError, Clock, estimate
from .audio import read_wav, render

__all__ = ["AlignmentError", "Clock", "estimate", "read_wav", "render"]
