import asyncio
import os
import json
from pathlib import Path
from collections import defaultdict
from utils.prompt import SYSTEM_PROMPT, USER_TMPL
import aiofiles
from tqdm import tqdm
from json_repair import repair_json
import argparse

# -------------------------------------------
# OpenAI client (or your custom client)
# -------------------------------------------
from openai import AsyncOpenAI

# ------------------ 配置部分 ------------------
API_KEY = "sk-****"  # Replace with your actual key or load securely
BASE_URL = "https://api.deepseek.com/v1"  # Verify/adjust if using a different provider or endpoint
MODEL = "deepseek-chat"

MAX_CONCURRENCY = 700

OUT_SUFFIX = "_evaluated_by_llm"
LOG_SUFFIX = "_evaluation_log.jsonl"
SUMMARY_SUFFIX = "_summary.json"

# Tags used by the LLM Grader to determine correctness for summary statistics
CORRECT_TAGS_FOR_LLM_GRADER = {"CORRECT"}


# ------------------ OpenAI Client Setup ------------------
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=60
)


# ------------------ Utility Functions ------------------
def dbg(msg):
    tqdm.write(str(msg))


async def load_json_with_repair(path: Path):
    async with aiofiles.open(path, 'r', encoding='utf-8') as f:
        text = await f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        dbg(f"Attempting to repair JSON for file: {path.name}")
        try:
            fixed_json_text = repair_json(text)
            return json.loads(fixed_json_text)
        except Exception as e:
            dbg(f"Failed to load or repair JSON for {path.name}: {e}")
            raise


async def write_json(path: Path, data):
    async with aiofiles.open(path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(data, ensure_ascii=False, indent=2))


# ------------------ Candidate Answer Extraction Logic ------------------
def extract_candidate_answer(record: dict) -> str:
    model_eval_res = record.get("model_evaluation_result")
    candidate = ""

    if isinstance(model_eval_res, dict):
        raw_response = model_eval_res.get("model_raw_response")
        if raw_response is not None:
            if isinstance(raw_response, str):
                raw_response_str = raw_response.strip()
                if not raw_response_str:
                    pass
                elif (raw_response_str.startswith('{') and raw_response_str.endswith('}')) or \
                        (raw_response_str.startswith('[') and raw_response_str.endswith(']')):
                    try:
                        repaired = repair_json(raw_response_str, return_objects=False)
                        parsed = json.loads(repaired)
                        if isinstance(parsed, dict):
                            if "answer" in parsed:
                                candidate = str(parsed["answer"] if parsed["answer"] is not None else "").strip()
                            elif "model_answer" in parsed:
                                candidate = str(
                                    parsed["model_answer"] if parsed["model_answer"] is not None else "").strip()
                            else:
                                candidate = repaired
                        elif isinstance(parsed, str):
                            candidate = parsed.strip()
                        else:
                            candidate = json.dumps(parsed, ensure_ascii=False)
                    except Exception:
                        candidate = raw_response_str
                else:
                    candidate = raw_response_str
            elif isinstance(raw_response, dict):
                if "answer" in raw_response:
                    candidate = str(raw_response["answer"] if raw_response["answer"] is not None else "").strip()
                elif "model_answer" in raw_response:
                    candidate = str(
                        raw_response["model_answer"] if raw_response["model_answer"] is not None else "").strip()
                else:
                    candidate = json.dumps(raw_response, ensure_ascii=False)
            else:
                candidate = json.dumps(raw_response, ensure_ascii=False)

        if not candidate and candidate != "":
            model_answer = model_eval_res.get("model_answer")
            if model_answer is not None:
                candidate = str(model_answer).strip()

    if not candidate and candidate != "":
        if record.get("model_answer") is not None:
            candidate = str(record["model_answer"]).strip()
        else:
            raw_fallback = record.get("model_raw_response")
            if raw_fallback is not None:
                if isinstance(raw_fallback, str):
                    raw_fallback_str = raw_fallback.strip()
                    if not raw_fallback_str:
                        pass
                    elif (raw_fallback_str.startswith('{') and raw_fallback_str.endswith('}')) or \
                            (raw_fallback_str.startswith('[') and raw_fallback_str.endswith(']')):
                        try:
                            repaired_fallback = repair_json(raw_fallback_str, return_objects=False)
                            parsed_fallback = json.loads(repaired_fallback)
                            if isinstance(parsed_fallback, dict) and "answer" in parsed_fallback:
                                candidate = str(
                                    parsed_fallback["answer"] if parsed_fallback["answer"] is not None else "").strip()
                            elif isinstance(parsed_fallback, str):
                                candidate = parsed_fallback.strip()
                            else:
                                candidate = json.dumps(parsed_fallback, ensure_ascii=False)
                        except Exception:
                            candidate = raw_fallback_str
                    else:
                        candidate = raw_fallback_str
                elif isinstance(raw_fallback, dict):
                    if "answer" in raw_fallback:
                        candidate = str(raw_fallback["answer"] if raw_fallback["answer"] is not None else "").strip()
                    else:
                        candidate = json.dumps(raw_fallback, ensure_ascii=False)
                else:
                    candidate = json.dumps(raw_fallback, ensure_ascii=False)
            else:
                candidate = ""
    return candidate.strip()


# ------------------ LLM Grading Logic ------------------
async def grade_one(processing_qid: str, question: str, gold_answer: str, candidate_ans: str) -> dict:
    user_prompt = USER_TMPL.format(
        question=question or "<NO QUESTION PROVIDED>",
        gold_answer=gold_answer or "<NO GOLD ANSWER PROVIDED>",
        cand=candidate_ans or "<EMPTY CANDIDATE ANSWER>",
        qid=processing_qid
    )
    # This 'grader_qid' is the QID this script instance is currently processing.
    # It's used for logging and matching, especially if the original record had no QID.
    # The LLM is also asked to echo a 'qid' in its response.
    evaluation_result = {
        "script_processing_qid": processing_qid,  # QID used by this script for this item
        "llm_grader_input_prompt_user": user_prompt  # For debugging
    }
    llm_response_obj = None

    for attempt in range(3):
        try:
            llm_response_obj = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            break  # Success
        except Exception as e:
            dbg(f"API Error (Attempt {attempt + 1}/3) for QID={processing_qid}: {e}")
            if attempt < 2:
                await asyncio.sleep(1 + attempt * 2)
            else:
                evaluation_result["llm_grader_category"] = "API_ERROR"
                evaluation_result["llm_grader_explanation"] = f"API request failed after 3 retries: {str(e)}"
                return evaluation_result

    if not llm_response_obj or not llm_response_obj.choices or not llm_response_obj.choices[0].message or not \
    llm_response_obj.choices[0].message.content:
        dbg(f"LLM Grader returned empty response for QID={processing_qid}")
        evaluation_result["llm_grader_category"] = "GRADER_EMPTY_RESPONSE"
        evaluation_result["llm_grader_explanation"] = "LLM grader returned an empty or malformed response."
        return evaluation_result

    raw_llm_content_str = llm_response_obj.choices[0].message.content
    evaluation_result["llm_grader_raw_response"] = raw_llm_content_str

    try:
        repaired_json_str = repair_json(raw_llm_content_str)
        # content_dict is what the LLM returned, expected to have 'qid', 'category', 'explanation'
        content_dict_from_llm = json.loads(repaired_json_str)

        if not isinstance(content_dict_from_llm, dict):
            raise ValueError("LLM response, after repair and parsing, was not a dictionary.")

        evaluation_result["llm_grader_repaired_and_parsed_response"] = content_dict_from_llm

        # Extract fields from LLM's response and map them to our desired keys
        evaluation_result["llm_echoed_qid"] = content_dict_from_llm.get("qid", processing_qid)  # LLM should echo QID
        evaluation_result["llm_grader_category"] = content_dict_from_llm.get("category",
                                                                             "GRADER_MISSING_CATEGORY_FIELD")
        evaluation_result["llm_grader_explanation"] = content_dict_from_llm.get("explanation",
                                                                                "No explanation provided by LLM grader.")

        # Check for missing essential fields from LLM output
        if evaluation_result["llm_grader_category"] == "GRADER_MISSING_CATEGORY_FIELD" and \
                "explanation" not in content_dict_from_llm:  # if LLM also missed explanation
            evaluation_result["llm_grader_explanation"] = "LLM response missing 'category' and 'explanation' fields."
        elif evaluation_result["llm_grader_category"] == "GRADER_MISSING_CATEGORY_FIELD":
            evaluation_result[
                "llm_grader_explanation"] = f"LLM response missing 'category' field. LLM Explanation (if any): {evaluation_result['llm_grader_explanation']}"


    except Exception as e:
        dbg(f"Failed to parse LLM grader response for QID={processing_qid}: {e}. Raw: {raw_llm_content_str[:200]}...")
        evaluation_result["llm_grader_category"] = "GRADER_INVALID_FORMAT"
        evaluation_result[
            "llm_grader_explanation"] = f"LLM response parsing error: {str(e)}. Raw content: {raw_llm_content_str[:100]}..."
        # llm_echoed_qid might not be available if parsing failed early
        evaluation_result.setdefault("llm_echoed_qid", processing_qid)

    if evaluation_result.get("llm_echoed_qid") != processing_qid:
        dbg(f"QID MISMATCH: Script processed QID '{processing_qid}', LLM returned QID '{evaluation_result.get('llm_echoed_qid')}'. Check 'llm_echoed_qid'.")

    return evaluation_result


# ------------------ File Processing Logic ------------------
async def process_one_file(input_file_path_str: str, root_dir_path_obj: Path, eval_output_root_path_obj: Path):
    input_file_path = Path(input_file_path_str)

    try:
        relative_path_from_root = input_file_path.relative_to(root_dir_path_obj)

        q_type_folder_name = relative_path_from_root.parts[0]
        model_name_folder = relative_path_from_root.parts[1]
        original_file_full_name = relative_path_from_root.parts[5]
        original_file_stem = Path(original_file_full_name).stem
        original_file_suffix = Path(original_file_full_name).suffix

    except ValueError:
        dbg(f"Skipping file {input_file_path} as it's not under the specified ROOT_DIR {root_dir_path_obj}.")
        return

    base_output_dir_for_this_file = eval_output_root_path_obj / original_file_stem
    summary_specific_output_dir = base_output_dir_for_this_file / "summary"

    try:
        base_output_dir_for_this_file.mkdir(parents=True, exist_ok=True)
        summary_specific_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        dbg(f"Error creating output directories for '{original_file_stem}' (Type: {q_type_folder_name}, Model: {model_name_folder}): {e}")
        return

    output_json_path = base_output_dir_for_this_file / (original_file_stem + OUT_SUFFIX + original_file_suffix)
    log_jsonl_path = base_output_dir_for_this_file / (original_file_stem + LOG_SUFFIX)
    summary_json_path = summary_specific_output_dir / (original_file_stem + SUMMARY_SUFFIX)

    try:
        original_data_content = await load_json_with_repair(input_file_path)
    except Exception as e:
        dbg(f"CRITICAL: Failed to load/repair source file {input_file_path.name}. Skipping. Error: {e}")
        return

    is_list_input = isinstance(original_data_content, list)
    original_records_list = original_data_content if is_list_input else [original_data_content]

    evaluations_from_log = {}  # Stores full evaluation objects from log
    if log_jsonl_path.exists():
        dbg(f"Log file found for '{original_file_stem}' (Type: {q_type_folder_name}, Model: {model_name_folder}). Loading from: {log_jsonl_path.name}")
        try:
            async with aiofiles.open(log_jsonl_path, "r", encoding="utf-8") as log_f:
                line_num = 0
                async for line in log_f:
                    line_num += 1
                    line_content = line.strip()
                    if not line_content: continue
                    try:
                        # Each line in log is a full evaluation object as returned by grade_one
                        log_eval_object = json.loads(line_content)
                        # Use 'script_processing_qid' from log for matching
                        logged_script_qid = log_eval_object.get("script_processing_qid")
                        if logged_script_qid:
                            evaluations_from_log[logged_script_qid] = log_eval_object
                        else:
                            dbg(f"Warning: Log entry in {log_jsonl_path.name} (line ~{line_num}) missing 'script_processing_qid'. Content: {line_content[:100]}...")
                    except json.JSONDecodeError as e_parse:
                        dbg(f"Warning: JSON parse error in log {log_jsonl_path.name} (line ~{line_num}): {e_parse}. Content: {line_content[:100]}...")
        except Exception as e_read_log:
            dbg(f"Warning: Could not fully read log file {log_jsonl_path.name}. Processing records not found. Error: {e_read_log}")

    tasks_for_llm_grading = []
    final_output_records_list = []
    stats_counter = defaultdict(int)  # For summary statistics

    # This function updates statistics based on the LLM grader's category
    def update_llm_grader_statistics(llm_grader_assigned_category: str):
        stats_counter["TOTAL_SUCCESSFULLY_GRADED_ITEMS"] += 1
        stats_counter[llm_grader_assigned_category] += 1  # Count per LLM grader category
        if llm_grader_assigned_category in CORRECT_TAGS_FOR_LLM_GRADER:
            stats_counter["LLM_GRADED_HITS"] += 1

    for idx, current_record_dict in enumerate(original_records_list):
        if not isinstance(current_record_dict, dict):
            dbg(f"Warning: Item at index {idx} in '{original_file_full_name}' (Type: {q_type_folder_name}, Model: {model_name_folder}) is not a dict. Skipping.")
            if is_list_input:
                final_output_records_list.append(
                    {"error": "Invalid item format, not a dictionary", "original_index": idx,
                     "original_content": str(current_record_dict)[:200]})
            continue

        # This is the QID used by the script to track this item for grading.
        script_instance_processing_qid = current_record_dict.get("qid")
        if not script_instance_processing_qid:
            script_instance_processing_qid = f"{original_file_stem}_{model_name_folder}_autogen_{idx}"
            current_record_dict[
                "qid_autogenerated_for_grading"] = script_instance_processing_qid  # Mark if QID was generated

        final_output_records_list.append(current_record_dict)

        if script_instance_processing_qid in evaluations_from_log:
            # The full evaluation object from the log is attached.
            # Its internal key 'llm_grader_category' holds the LLM's assessment.
            llm_evaluation_from_log = evaluations_from_log[script_instance_processing_qid]
            current_record_dict["llm_evaluation"] = llm_evaluation_from_log

            category_assigned_by_llm_in_log = llm_evaluation_from_log.get("llm_grader_category", "UNKNOWN_FROM_LOG")

            if category_assigned_by_llm_in_log != "API_ERROR":
                update_llm_grader_statistics(category_assigned_by_llm_in_log)
            # else: API_ERROR from log, already handled by not writing to log if we want auto-retry.
            # The current logic means if API_ERROR is in log, it's considered 'processed' for this run.
        else:
            tasks_for_llm_grading.append(
                (current_record_dict, script_instance_processing_qid)
            )

    progress_bar_desc = f"Grading {q_type_folder_name}/{model_name_folder}/{original_file_stem}"
    if not tasks_for_llm_grading:
        dbg(f"No new records require LLM grading for '{original_file_full_name}' (Type: {q_type_folder_name}, Model: {model_name_folder}).")
    else:
        dbg(f"LLM Grading {len(tasks_for_llm_grading)} new/pending records for '{original_file_full_name}' (Type: {q_type_folder_name}, Model: {model_name_folder}).")

        async with aiofiles.open(log_jsonl_path, 'a', encoding='utf-8') as log_f_append:
            pbar = tqdm(total=len(tasks_for_llm_grading), desc=progress_bar_desc, unit="item", leave=False)
            semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

            async def evaluate_and_log_one_record(record_ref_to_update: dict, qid_for_grading_script: str):
                nonlocal pbar
                async with semaphore:
                    question_text = record_ref_to_update.get("question", "")
                    gold_answer_text = record_ref_to_update.get("gold_answer", "")
                    candidate_answer = extract_candidate_answer(record_ref_to_update)

                    # llm_full_evaluation_result is the dict returned by grade_one
                    llm_full_evaluation_result = await grade_one(qid_for_grading_script, question_text,
                                                                 gold_answer_text, candidate_answer)

                    # Attach the entire LLM evaluation object under "llm_evaluation" key
                    record_ref_to_update["llm_evaluation"] = llm_full_evaluation_result

                    # Get the category assigned by the LLM grader for statistics and logging decisions
                    llm_assigned_category = llm_full_evaluation_result.get("llm_grader_category", "UNKNOWN_AFTER_LLM")

                    if llm_assigned_category != "API_ERROR":
                        update_llm_grader_statistics(llm_assigned_category)
                        # Write the full evaluation object (returned by grade_one) to log
                        log_entry_str = json.dumps(llm_full_evaluation_result, ensure_ascii=False) + "\n"
                        await log_f_append.write(log_entry_str)
                        await log_f_append.flush()
                    else:
                        dbg(f"QID {qid_for_grading_script} resulted in API_ERROR. Not writing to log for auto-retry on next run.")

                    pbar.update(1)

            await asyncio.gather(*(
                evaluate_and_log_one_record(record_dict_ref, script_qid) for record_dict_ref, script_qid in
            tasks_for_llm_grading
            ))
            pbar.close()

    final_json_to_write = final_output_records_list
    if not is_list_input:
        final_json_to_write = final_output_records_list[0] if final_output_records_list else {}

    await write_json(output_json_path, final_json_to_write)

    # Summary statistics are based on 'llm_grader_category'
    total_items_successfully_graded = stats_counter["TOTAL_SUCCESSFULLY_GRADED_ITEMS"]
    llm_graded_correct_hits = stats_counter["LLM_GRADED_HITS"]

    grader_accuracy = round(llm_graded_correct_hits / total_items_successfully_graded,
                            4) if total_items_successfully_graded > 0 else 0.0

    api_error_count_this_run = 0
    for record in final_output_records_list:
        # Check the "llm_evaluation" field and its "llm_grader_category"
        if record.get("llm_evaluation", {}).get("llm_grader_category") == "API_ERROR":
            api_error_count_this_run += 1

    summary_data = {
        "source_input_file_full_name": original_file_full_name,
        "source_input_file_stem": original_file_stem,
        "question_type_folder": q_type_folder_name,
        "model_name_from_path": model_name_folder,
        "total_items_in_source_file": len(original_records_list),
        "total_items_successfully_graded_by_llm_grader_this_run": total_items_successfully_graded,
        "items_with_api_error_this_run": api_error_count_this_run,
        "items_graded_as_correct_by_llm_grader": llm_graded_correct_hits,
        "accuracy_according_to_llm_grader_on_successful_grades": grader_accuracy,
        "llm_grader_category_breakdown_successful_grades": {
            cat: count for cat, count in stats_counter.items()
            if cat not in ("TOTAL_SUCCESSFULLY_GRADED_ITEMS", "LLM_GRADED_HITS")
        }
    }
    await write_json(summary_json_path, summary_data)
    dbg(f"Finished: {progress_bar_desc}. LLM Grader Accuracy (successful grades): {grader_accuracy * 100:.2f}%. Summary: {summary_json_path.name}")


# ------------------ Main Orchestration Logic ------------------
async def main(args):
    eval_path = args.eval_path
    out_path = args.out_path
    model_name = args.model
    question_type = args.question_type
    temperature = args.temperature
    top_p = args.top_p
    ROOT_DIR = os.path.join(eval_path, question_type, model_name, f'tem{temperature}', f'top_k{top_p}', 'evaluation')
    EVALUATED_OUTPUT_ROOT_DIR = os.path.join(out_path, question_type, model_name, f'tem{temperature}', f'top_k{top_p}')
    root_dir = Path(ROOT_DIR)
    eval_output_root_dir = Path(EVALUATED_OUTPUT_ROOT_DIR)

    if not root_dir.is_dir():
        print(f"ERROR: Input ROOT_DIR '{ROOT_DIR}' does not exist or is not a directory.")
        return
    if not EVALUATED_OUTPUT_ROOT_DIR:
        print(f"ERROR: EVALUATED_OUTPUT_ROOT_DIR is not set. Please specify a path.")
        return
    if root_dir.resolve() == eval_output_root_dir.resolve():
        print(f"CRITICAL ERROR: ROOT_DIR and EVALUATED_OUTPUT_ROOT_DIR must be different paths.")
        return

    try:
        eval_output_root_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create EVALUATED_OUTPUT_ROOT_DIR '{eval_output_root_dir}': {e}")
        return

    json_files_to_process = []
    for json_file in root_dir.glob("*.json"):
        if json_file.is_file():
            json_files_to_process.append(json_file)

    if not json_files_to_process:
        print(f"No JSON files found matching the expected structure: {ROOT_DIR}/<Q_Type>/<Model_Name>/*.json")
        return

    print(f"Found {len(json_files_to_process)} JSON files to process from '{ROOT_DIR}'.")
    print(f"Outputs will be saved under '{eval_output_root_dir}'.")

    for file_path_obj in tqdm(json_files_to_process, desc="Overall Progress (Files)", unit="file"):
        await process_one_file(str(file_path_obj), eval_path, eval_output_root_dir)

    print("\nAll processing finished.")
    print(f"Check '{eval_output_root_dir}' for evaluation outputs, logs, and summaries.")

parser = argparse.ArgumentParser()
if __name__ == "__main__":
    parser.add_argument('--eval_path', default='./result', type=str, help='Directory containing answers LLM generated')
    parser.add_argument('--out_path', default='./eval', type=str, help='Directory saving the evaluation result')
    parser.add_argument('--model', default='deepseek-chat', type=str, help='Name of LLM model')
    parser.add_argument('--question_type', default='tf', type=str, help='Type of chioce: (fill, general, multiple, numerical, proof, single, table, tf)')
    parser.add_argument('--temperature', default=0.2, type=float, help='temperature of the LLM')
    parser.add_argument('--top_p', default=0.95, type=float, help='top of the LLM')
    args = parser.parse_args()
    asyncio.run(main(args))