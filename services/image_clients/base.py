"""
Shared protocol for all image clients. Each provider (Midjourney, Grok Imagine,
Z-Image Base, Z-Image-Turbo) implements .generate(prompt, n) and returns a list
of file paths to the generated PNGs.
"""

from typing import List, Protocol


class ImageClient(Protocol):
    """Minimum surface every image-generation client must implement."""

    def generate(self, prompt: str, n: int = 1, aspect_ratio: str = "16:9") -> List[str]:
        """Generate *n* images for *prompt*. Return a list of saved file paths."""
        ...
