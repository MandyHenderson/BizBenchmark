"""dataloader.py

Utilities for loading question and answer data from JSON files.

This module provides two helper functions:
    - load_questions: Parse a question JSON file and prepare records for evaluation.
    - load_answers_path: Recursively collect answer JSON files from a directory tree.

The functions implement resume‑from‑checkpoint logic, robust file‑encoding
fallback, and defensive error handling to make batch processing resilient.
"""

import json
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from json import JSONDecodeError

from utils.utils import collect_questions


def load_questions(
    input_json_file_path: Path,
    eval_jsonl_path: Path,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    """Load questions from a single JSON file and apply resume logic.

    The function scans *input_json_file_path* (which may contain either a
    single root object or a list of objects) and extracts every nested
    question item via ``utils.utils.collect_questions``. It then compares the
    extracted question IDs (``qid`` fields) with the IDs already present in
    *eval_jsonl_path* so that previously processed questions are skipped when
    the script is re‑run.

    Args:
        input_json_file_path (Path): Path to the source question JSON file.
        eval_jsonl_path (Path): Path to the evaluation ``.jsonl`` file. If the
            file exists it is parsed to determine which questions have already
            been handled.

    Returns:
        Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]
            A two‑tuple where
            - all_records_for_this_file: list that begins with any previously
              processed records loaded from *eval_jsonl_path*. Callers are
              expected to append new records to the same list before writing
              it back to disk in a single operation.
            - questions_to_process_now: list of new question objects that have
              not yet been processed. ``None`` is returned for both elements
              if the input file could not be read or contained no valid
              questions.

    Notes:
        * The function prints progress information instead of raising
          exceptions so that a batch runner can continue with the next file.
        * The resume mechanism relies on each question record containing a
          non‑empty ``qid`` string.

    Example:
        >>> all_prev, pending = load_questions(
        ...     Path('sample_questions.json'),
        ...     Path('eval_records.jsonl'),
        ... )
        >>> for q in pending:
        ...     process_question(q)
    """
    # Attempt to read file content using UTF‑8 with and without BOM.
    try:
        try:
            file_content = input_json_file_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            print(
                f"    [Info] Failed to decode {input_json_file_path.name} using "
                "utf-8-sig, trying utf-8."
            )
            file_content = input_json_file_path.read_text(encoding="utf-8")

        data_from_file_raw = json.loads(file_content)

    except FileNotFoundError:
        print(f"    [Error] File not found: {input_json_file_path}. Skipping.")
        return None, None
    except JSONDecodeError as e:
        print(f"    [Error] Invalid JSON in {input_json_file_path.name}: {e}. Skipping.")
        return None, None
    except Exception as e:
        print(f"    [Error] Failed to read {input_json_file_path.name}: {e}. Skipping.")
        return None, None

    # Normalise to list so collect_questions can work uniformly.
    data_to_scan: List[Any] = (
        data_from_file_raw if isinstance(data_from_file_raw, list) else [data_from_file_raw]
    )

    all_question_objects_from_file: List[Dict[str, Any]] = []
    collect_questions(data_to_scan, all_question_objects_from_file)

    if not all_question_objects_from_file:
        print(f"    [Info] No valid question items found in {input_json_file_path.name}. Skipping.")
        return None, None

    print(
        f"    Found {len(all_question_objects_from_file)} valid question items "
        f"in {input_json_file_path.name}."
    )

    # Initialise containers for resume logic.
    previously_processed_records: List[Dict[str, Any]] = []
    processed_qids: set[str] = set()  # QIDs that have already been processed.

    # ------------------------------------------------------------------
    # Resume from existing evaluation file, if any.
    # ------------------------------------------------------------------
    if eval_jsonl_path.exists():
        print(f"    [Info] Resuming from existing file: {eval_jsonl_path.name}")
        try:
            with eval_jsonl_path.open("r", encoding="utf-8") as f_jsonl:
                for line_num, line in enumerate(f_jsonl):
                    try:
                        record = json.loads(line)
                        qid = record.get("qid")

                        # Only add well‑formed, non‑empty QIDs to the skip list.
                        if isinstance(qid, str) and qid.strip():
                            previously_processed_records.append(record)
                            processed_qids.add(qid)
                        else:
                            # Keep the record for aggregation, but we cannot use
                            # it to decide skipping logic.
                            previously_processed_records.append(record)

                    except json.JSONDecodeError:
                        print(
                            f"    [Warning] Skipped malformed JSON line "
                            f"{line_num + 1} in {eval_jsonl_path.name}"
                        )

            # 检查文件完整性：如果已处理的问题数等于源文件问题数，才完全跳过
            total_source_questions = len(all_question_objects_from_file)
            total_processed_questions = len(processed_qids)
            
            print(
                f"    Loaded {len(previously_processed_records)} records from "
                f"{eval_jsonl_path.name} "
                f"({total_processed_questions} unique valid QIDs used for resume)."
            )
            
            # 完整性检查
            if total_processed_questions == total_source_questions:
                print(f"    ✅ 文件已完成处理 ({total_processed_questions}/{total_source_questions})")
            else:
                print(f"    ⚠️  文件未完成处理 ({total_processed_questions}/{total_source_questions})，将继续处理剩余问题")
        except Exception as e:
            print(
                f"    [Error] Failed to read/parse {eval_jsonl_path.name} for "
                f"resuming: {e}. Treating all items as new."
            )
            previously_processed_records = []
            processed_qids = set()

    # ------------------------------------------------------------------
    # Determine which questions still need processing.
    # ------------------------------------------------------------------
    questions_to_process_now: List[Dict[str, Any]] = []
    for qobj in all_question_objects_from_file:
        qid = qobj.get("qid")

        # Process item if:
        #   1. qid is not a non‑empty string, or
        #   2. qid not found in the set of processed IDs.
        if not (isinstance(qid, str) and qid.strip() and qid in processed_qids):
            questions_to_process_now.append(qobj)

    num_skipped = len(all_question_objects_from_file) - len(questions_to_process_now)
    if num_skipped > 0:
        print(
            f"    Skipped {num_skipped} already processed questions "
            f"(matched QIDs found in {eval_jsonl_path.name})."
        )

    # Aggregate previously processed and yet‑to‑process records so that callers
    # can write everything back in one go.
    all_records_for_this_file: List[Dict[str, Any]] = list(previously_processed_records)

    return all_records_for_this_file, questions_to_process_now


def load_answers_path(input_dir: Path) -> List[Path]:
    """Recursively collect all ``.json`` answer files under *input_dir*.

    Expected directory structure::

        <input_dir>/
            <question_type>/
                <model_name>/
                    answers.json

    Every ``*.json`` file encountered at the third level will be returned.

    Args:
        input_dir (Path): Root directory containing sub‑directories arranged
            by question type and model name.

    Returns:
        List[Path]: Absolute paths to every answer JSON file discovered.
    """
    json_files_to_process: List[Path] = []

    # Walk the directory tree up to three levels.
    for q_type_dir in input_dir.iterdir():
        if q_type_dir.is_dir():
            for model_name_dir in q_type_dir.iterdir():
                if model_name_dir.is_dir():
                    # Pick up *.json files directly under the model directory.
                    for json_file in model_name_dir.glob("*.json"):
                        if json_file.is_file():
                            json_files_to_process.append(json_file)

    return json_files_to_process
