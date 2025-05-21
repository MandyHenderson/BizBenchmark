import re
from typing import Any, Dict, List, Tuple

class UpstreamSaturationError(Exception):
    pass

def split_options(text: str) -> Tuple[str, List[str]]:
    OPTION_RE = re.compile(r"(?m)^\s*([A-D])\s*[).．、:：]\s*(.*?)(?=^\s*[A-D]\s*[).．、:：]|\Z)")
    matches = list(OPTION_RE.finditer(text))
    if not matches:
        return text.strip(), []
    first_match_start = matches[0].start()
    stem = text[:first_match_start].strip()
    opts = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        opts.append(text[start:end].strip())
    return stem, opts

def collect_questions(container: Any, bag: List[Dict[str, Any]]):
    if isinstance(container, dict):
        if container.get("question") and isinstance(container["question"], str):
            bag.append(container)
        else:
            # Recursively check values if the current dict itself isn't a direct question item
            for v in container.values():
                collect_questions(v, bag)
    elif isinstance(container, list):
        for item in container:
            collect_questions(item, bag)