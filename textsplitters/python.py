from __future__ import annotations

from typing import Any
from textsplitters.base import Language
from textsplitters.textsplitter import TextSplitter


class PythonTextSplitter(TextSplitter):
    """Attempts to split the text along Python syntax."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a PythonCodeTextSplitter."""
        separators = self.get_separators_for_language(Language.PYTHON)
        super().__init__(separators=separators, **kwargs)
