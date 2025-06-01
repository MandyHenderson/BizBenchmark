"""utils.py

Shared utility helpers for the evaluation pipeline.

Contents
--------
- UpstreamSaturationError – custom exception signalling provider saturation
- split_options           – split raw question text into stem and option list
- collect_questions       – recursively collect question objects from nested
                             JSON‑like structures
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


class UpstreamSaturationError(Exception):
    """Raised when the upstream LLM or proxy reports saturation.

    Pipeline code catches this exception to abort the current batch early
    instead of retrying indefinitely.
    """


def split_options(text: str) -> Tuple[str, List[str]]:
    """Return the stem and a list of option strings found in *text*.

    The function searches for lines that begin with a label ``A``–``D``
    followed by one of ``)``, ``.``, ``．``, ``、``, ``:`` or ``：`` in either
    English or Chinese punctuation. Matching is multiline (`re.M`) and
    non‑greedy so that each option captures everything up to the next label or
    end of string.

    Args:
        text: Raw question text that may contain in‑line options.

    Returns:
        Tuple where
            - element 0 is the stem string (question without options)
            - element 1 is a list of option strings in their original order
    """
    option_re = re.compile(r"(?m)^\s*([A-D])\s*[).．、:：]\s*(.*?)(?=^\s*[A-D]\s*[).．、:：]|\Z)")
    matches = list(option_re.finditer(text))
    if not matches:
        return text.strip(), []

    first_match_start = matches[0].start()
    stem = text[:first_match_start].strip()

    opts: List[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        opts.append(text[start:end].strip())

    return stem, opts


def collect_questions(container: Any, bag: List[Dict[str, Any]]) -> None:
    """Recursively append all dicts that look like question items to *bag*.

    A *question item* is any dict that has a non‑empty ``question`` key whose
    value is a string. Nested structures (lists or dicts) are traversed
    depth‑first.

    Args:
        container: Current JSON fragment (dict or list) to scan.
        bag: Accumulator list where discovered question dicts are appended.
    """
    if isinstance(container, dict):
        if container.get("question") and isinstance(container["question"], str):
            bag.append(container)
        else:
            # Recursively inspect child values.
            for value in container.values():
                collect_questions(value, bag)
    elif isinstance(container, list):
        for item in container:
            collect_questions(item, bag)
