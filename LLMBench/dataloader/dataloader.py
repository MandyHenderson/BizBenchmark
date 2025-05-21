import json
import os
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from json import JSONDecodeError
from utils.utils import collect_questions
def load_questions(input_json_file_path, eval_jsonl_path):
    eval_jsonl_path = Path(eval_jsonl_path)
    try:
        try:
            file_content = input_json_file_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            print(f"    [Info] Failed to decode {input_json_file_path.name} using utf-8-sig, trying utf-8.")
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

    # Ensure data_to_scan is a list for collect_questions
    data_to_scan = data_from_file_raw if isinstance(data_from_file_raw, list) else [data_from_file_raw]
    all_question_objects_from_file: List[Dict[str, Any]] = []
    collect_questions(data_to_scan, all_question_objects_from_file)

    if not all_question_objects_from_file:
        print(f"    [Info] No valid question items found in {input_json_file_path.name}. Skipping.")
        return None, None

    print(
        f"    Found {len(all_question_objects_from_file)} valid question items in {input_json_file_path.name}.")

    previously_processed_records: List[Dict[str, Any]] = []
    processed_qids: set[str] = set()  # Set of QIDs that have been successfully processed and recorded

    if eval_jsonl_path.exists():
        print(f"    [Info] Resuming from existing file: {eval_jsonl_path.name}")
        try:
            with eval_jsonl_path.open("r", encoding="utf-8") as f_jsonl:
                for line_num, line in enumerate(f_jsonl):
                    try:
                        record = json.loads(line)
                        qid = record.get("qid")
                        # Add to processed_qids only if qid is valid string.
                        # Records without valid QID can't be reliably skipped for resume.
                        if isinstance(qid, str) and qid.strip():  # Ensure QID is a non-empty string
                            previously_processed_records.append(record)
                            processed_qids.add(qid)
                        else:
                            # If record has no valid QID, still load it for aggregation but don't add to processed_qids for skipping.
                            previously_processed_records.append(record)
                            # print(f"    [Warning] Line {line_num + 1} in {eval_jsonl_path.name} is missing valid 'qid'. Loaded but cannot be skipped in resume.")
                    except json.JSONDecodeError:
                        print(f"    [Warning] Skipped malformed JSON line {line_num + 1} in {eval_jsonl_path.name}")
            print(
                f"    Loaded {len(previously_processed_records)} records from {eval_jsonl_path.name} ({len(processed_qids)} unique valid QIDs used for resume).")
        except Exception as e:
            print(f"    [Error] Failed to read/parse {eval_jsonl_path.name} for resuming: {e}. Treating all items as new.")
            previously_processed_records = []
            processed_qids = set()

    questions_to_process_now: List[Dict[str, Any]] = []
    for qobj in all_question_objects_from_file:
        qid = qobj.get("qid")
        # If qid is not a non-empty string OR qid is not in processed_qids set, then process it.
        if not (isinstance(qid, str) and qid.strip() and qid in processed_qids):
            questions_to_process_now.append(qobj)
        # else: item has a valid QID and it's in processed_qids, so skip.

    num_skipped = len(all_question_objects_from_file) - len(questions_to_process_now)
    if num_skipped > 0:
        print(f"    Skipped {num_skipped} already processed questions (matched QIDs found in {eval_jsonl_path.name}).")
    all_records_for_this_file: List[Dict[str, Any]] = list(previously_processed_records)

    return all_records_for_this_file, questions_to_process_now

def load_answers_path(input_dir):
    json_files_to_process = []
    for q_type_dir in input_dir.iterdir():
        if q_type_dir.is_dir():
            for model_name_dir in q_type_dir.iterdir():
                if model_name_dir.is_dir():
                    for json_file in model_name_dir.glob("*.json"):
                        if json_file.is_file():
                            json_files_to_process.append(json_file)
    return json_files_to_process

