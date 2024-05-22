
from __future__ import annotations

from typing import Any
from textsplitters.base import Language
from textsplitters.textsplitter import TextSplitter

EXT_TO_LANG = {'py': Language.PYTHON,
               'c': Language.C,
               'cpp': Language.CPP,
               'cs': Language.CSHARP,
               'go': Language.GO,
               'java': Language.JAVA,
               'js': Language.JS,
               'kt': Language.KOTLIN,
               'php': Language.PHP,
               'rb': Language.RUBY,
               'rs': Language.RUST,
               'scala': Language.SCALA,
               'swift': Language.SWIFT,
               'md': Language.MARKDOWN,
               'rst': Language.RST,
               'tex': Language.LATEX,
               'html': Language.HTML,
               'sol': Language.SOL,
               'cob': Language.COBOL,
               'lua': Language.LUA,
               'pl': Language.PERL,
               'ts': Language.TS,
               'proto': Language.PROTO,
               }


class CodeSplitter(TextSplitter):
    """Attempts to split the text along Python syntax."""

    def __init__(self, extension: str, **kwargs: Any) -> None:
        """Initialize a PythonCodeTextSplitter."""
        separators = self.get_separators_for_language(EXT_TO_LANG[extension])
        super().__init__(separators=separators, **kwargs)
