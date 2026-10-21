"""The optional provenance argument of add_memory, shared by both backends.

Kept free of heavy imports: the Zep backend uses it too.
"""

from __future__ import annotations

import json
from typing import Any


def normalize_source(source: Any) -> dict:
    """The optional provenance argument: a dict, a JSON object string, or prose."""
    if source is None or source == "":
        return {}
    if isinstance(source, dict):
        return source
    if isinstance(source, str):
        try:
            v = json.loads(source)
        except ValueError:
            return {"note": source}
        return v if isinstance(v, dict) else {"note": v}
    return {"note": str(source)}
