from openai import OpenAI
from dataloader.dataloader import load_questions

import os
from tqdm import tqdm
import concurrent.futures
from typing import Any, Dict, List, Optional
import sys
import json
from utils.prompt import *
from json import JSONDecodeError
from utils.utils import *
from pathlib import Path
from json_repair import repair_json
# ------------------------ Configuration ------------------------ #
# !!! IMPORTANT: Replace with your actual API key and potentially adjust BASE_URL !!!
MAX_WORKERS = 1
MAX_LLM_RETRIES = 3
LLM_RETRY_DELAY = 10
MAX_TOKENS_FOR_JSON = 60
UPSTREAM_SATURATION_KEYWORD = "Saturation Detected"
API_KEY = "sk-****"  # Replace with your actual key or load securely
BASE_URL = "https://api.deepseek.com/v1"  # Verify/adjust if using a different provider or endpoint
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
def process_directory(input_dir, output_dir, question_type, model_name, temperature, top_p):
    input_dir = Path(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    MAX_WORKERS = 1
    print(f"output_dir: {output_dir}")
    print(f"model_name: {model_name}")
    print(f"temperature: {temperature}")
    print(f"top_p: {top_p}")
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    json_files_to_process = [f for f in input_dir.glob("*.json") if
                             f.is_file() and f.parent.resolve() == input_dir.resolve()]
    for input_json_file_path in json_files_to_process:
        eval_jsonl_path = os.path.join(eval_dir, f"{input_json_file_path.stem}_eval.jsonl")
        eval_jsonl_path = Path(eval_jsonl_path)
        eval_json_path = os.path.join(eval_dir, f"{input_json_file_path.stem}_eval.json")
        eval_json_path = Path(eval_json_path)
        previously_processed_records, questions_to_process_now = load_questions(input_json_file_path, eval_jsonl_path)
        all_records_for_this_file: List[Dict[str, Any]] = list(previously_processed_records)
        newly_processed_and_written_count = 0
        if questions_to_process_now:
            print(f"    Processing {len(questions_to_process_now)} new or reprocessable question items...")
            progress_bar: Optional[tqdm] = None
            future_to_qobj: Dict[concurrent.futures.Future, Dict[str, Any]] = {}
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    for qobj_to_submit in questions_to_process_now:
                        future = executor.submit(process_question_item, qobj_to_submit, model_name, question_type, temperature, top_p)
                        future_to_qobj[future] = qobj_to_submit

                    pbar_desc = f"LLM({model_name}|T:{temperature}|Top_p:{top_p}) {input_json_file_path.name[:15]}"
                    progress_bar = tqdm(concurrent.futures.as_completed(future_to_qobj), total=len(future_to_qobj),
                                        desc=pbar_desc, unit="q", ncols=120)

                    for future in progress_bar:
                        original_qobj_for_future = future_to_qobj[future]
                        qid_for_log = original_qobj_for_future.get("qid", "NO_QID_IN_ORIGINAL_FOR_FUTURE")

                        try:
                            result_record = future.result()

                            if result_record:
                                all_records_for_this_file.append(result_record)
                                with eval_jsonl_path.open("a", encoding="utf-8") as f_jsonl:
                                    f_jsonl.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                                newly_processed_and_written_count += 1

                        except UpstreamSaturationError as use:
                            if progress_bar: progress_bar.write(
                                f"\n[FATAL ERROR] Upstream saturation (QID: {qid_for_log}, Model: {model_name}, Temp: {temperature}, Top_p: {top_p}): {use}.")
                            executor.shutdown(wait=True, cancel_futures=True)
                            if progress_bar: progress_bar.close()
                            print("[INFO] Terminating due to upstream saturation.")
                            sys.exit(1)

                        except KeyboardInterrupt:
                            if progress_bar: progress_bar.write("\n[INFO] User interrupted during task result waiting.")
                            executor.shutdown(wait=True, cancel_futures=True)
                            if progress_bar: progress_bar.close()
                            print("[INFO] Processing stopped. Program will exit.")
                            sys.exit(0)

                        except Exception as exc_task:
                            if progress_bar: progress_bar.write(
                                f'[ERROR] QID "{qid_for_log}" (Model: {model_name}, Temp: {temperature}, Top_p: {top_p}) encountered an unexpected framework exception: {exc_task}.')
                            qid_from_original = original_qobj_for_future.get("qid")
                            if isinstance(qid_from_original, str) and qid_from_original.strip():
                                error_placeholder = {
                                    **original_qobj_for_future,
                                    "model_evaluation_result": {
                                        "model_raw_response": "",
                                        "model_answer": "Error_Framework_Exception",
                                        "error": f"Task framework exception: {str(exc_task)}"
                                    }
                                }
                                all_records_for_this_file.append(error_placeholder)
                                with eval_jsonl_path.open("a", encoding="utf-8") as f_jsonl:
                                    f_jsonl.write(json.dumps(error_placeholder, ensure_ascii=False) + "\n")
                                newly_processed_and_written_count += 1
                            else:
                                if progress_bar: progress_bar.write(
                                    f'[INFO] QID "{qid_for_log}" framework exception, but original item lacks valid QID; cannot reliably log placeholder.')

            except KeyboardInterrupt:
                print("\n[INFO] User interrupted during task submission/setup.")
                if progress_bar and not progress_bar.disable: progress_bar.close()
                print("[INFO] Processing stopped. User interrupt will cause program to exit.")
                sys.exit(0)
            except Exception as pool_exc:
                print(
                    f"[ERROR] Unexpected error during thread pool execution (Model: {model_name}, Temp: {temperature}, Top_p: {top_p}): {pool_exc}")
            finally:
                if progress_bar and not progress_bar.disable:
                    progress_bar.close()
        

            if newly_processed_and_written_count > 0:
                print(f"    LLM processing of new items completed. {newly_processed_and_written_count} results appended to {eval_jsonl_path.name}")
            elif questions_to_process_now:
                print(f"    LLM processing completed, but no items successfully produced results in {eval_jsonl_path.name} (possible API errors, skips, or data issues).")

        elif not previously_processed_records:
            print(f"    [INFO] {input_json_file_path.name} has no new items to process and no previous results found.")
        else:
            print(f"    [INFO] All items in {input_json_file_path.name} have been processed and loaded from {eval_jsonl_path.name}.")

        if not all_records_for_this_file:
            print(f"    [INFO] No records (new or old) to summarize or save for {input_json_file_path.name}.")
            continue

        # Deduplicate records based on QID, keeping the latest version for each
        final_unique_records_map = {}
        temp_id_counter = 0
        for rec in all_records_for_this_file:
            qid = rec.get("qid")
            if isinstance(qid, str) and qid.strip():
                final_unique_records_map[qid] = rec
            else:
                temp_key = f"__temp_no_qid_at_final_agg_{temp_id_counter}__"
                final_unique_records_map[temp_key] = rec
                # print(f"    [WARNING] Found a record still missing a valid QID during final aggregation...")
                temp_id_counter += 1
        final_records_list = list(final_unique_records_map.values())
        if final_records_list:  # Only write if there's something to write
            try:
                eval_json_path.write_text(json.dumps(final_records_list, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
                print(
                    f"    Merged evaluation data ({len(final_records_list)} unique/final records) has been saved to {eval_json_path.name}")
            except Exception as e:
                print(f"    [Error] Failed to write final merged JSON to {eval_json_path.name}: {e}")


        
def call_llm(qobj: Dict[str, Any], llm_model: str, question_type: str, temperature: float, top_p: float) -> Dict[str, Any]:
    global client
    if not client:
        raise ConnectionError("LLM client is not initialized. Please check your API key and base URL.")
    user_prompt_parts = [f"Question: {str(qobj.get('question')).strip()}"]
    if question_type == "single":
        raw_q_original = str(qobj.get("question")).strip()  # Keep original for record if stem changes
        raw_q_for_llm = raw_q_original  # This might change if options are parsed from it
        qid = qobj.get("qid")
        if not (isinstance(qid, str) and qid.strip()):  # If no valid QID from input
            qid = f"NO_QID_INPUT_{raw_q_original[:20]}"  # Create a temporary one for logging
        opts_list_input = qobj.get("options")
        context_str_bg = qobj.get("background_text")
        q_context_resolved = ""
        if isinstance(context_str_bg, str) and context_str_bg.strip():
            q_context_resolved = context_str_bg.strip()

        opts_list_processed: List[str] = []
        if isinstance(opts_list_input, list) and opts_list_input:
            opts_list_processed = [str(opt).strip() for opt in opts_list_input if
                                str(opt).strip()]  # Ensure options are strings and stripped
        else:  # Try to parse options from question text if "options" field is missing/empty
            processed_q_stem, parsed_opts = split_options(raw_q_original)
            if parsed_opts:
                raw_q_for_llm = processed_q_stem  # Use the stem as question if options were in it
                opts_list_processed = parsed_opts
        question, options = raw_q_for_llm, opts_list_processed
        if options:
            user_prompt_parts.append("Options:\n" + "\n".join(options))
        if q_context_resolved and q_context_resolved.strip():
            user_prompt_parts.append(f"\nContext:\n{q_context_resolved}")
    elif question_type == "multiple":
        raw_options = qobj.get("options")
        options_list: List[str] = []
        if isinstance(raw_options, list) and all(isinstance(opt, str) for opt in raw_options):
            options_list = raw_options
        actual_option_labels = []
        if options_list:
            options_str = "\n".join(options_list)
            user_prompt_parts.append(f"Options:\n{options_str}")
            for opt_str in options_list:
                if opt_str and len(opt_str) >= 1 and opt_str[0].isalpha() and (
                        len(opt_str) == 1 or opt_str[1] == ')' or opt_str[1] == '.'):  # Handles "A", "A)", "A."
                    actual_option_labels.append(opt_str[0])
    table_html = qobj.get("table_html")
    if table_html and str(table_html).strip():
        user_prompt_parts.append(f"\nTable (HTML):\n```html\n{str(table_html).strip()}\n```")

    formula_context = qobj.get("formula_context")
    if formula_context and str(formula_context).strip():
        user_prompt_parts.append(f"\nFormula Context:\n{str(formula_context).strip()}")

    question_context = qobj.get("question_context")  # General context
    if question_context and str(question_context).strip():
        user_prompt_parts.append(f"\nAdditional Context:\n{str(question_context).strip()}")

    user_prompt = "\n\n".join(user_prompt_parts)
    if question_type == "single":
        prompt = single_prompt
    elif question_type == "multiple":
        prompt = multiple_prompt
    elif question_type == "proof":
        prompt = proof_prompt
    elif question_type == "table":
        prompt = table_prompt
    elif question_type == "general":
        prompt = general_prompt
    elif question_type == "numerical":
        prompt = numerical_prompt
    elif question_type == "fill":
        prompt = fill_prompt
    elif question_type == "tf":
        prompt = tf_prompt

    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt}]

    raw_content: Optional[str] = None
    extracted_answer: str = "Error_LLM_Logic_Failed"
    err: Optional[str] = None
    parsed_json: Optional[Dict[str, Any]] = None

    try:
        rsp = client.chat.completions.create(
            model=llm_model, messages=messages, temperature=temperature, top_p=top_p,
            max_tokens=MAX_TOKENS_FOR_JSON, response_format={'type': 'json_object'}, stream=False,
            timeout=60.0
        )
        raw_content = rsp.choices[0].message.content if rsp.choices and rsp.choices[0].message else ""

        if raw_content and UPSTREAM_SATURATION_KEYWORD in raw_content:
            raise UpstreamSaturationError(f"Upstream saturation detected in response: {raw_content[:200]}...")

        if not raw_content:
            err = "Error: LLM returned empty content (API call succeeded)"
        else:
            try:
                parsed_json = json.loads(raw_content)
            except JSONDecodeError:
                try:
                    repaired_json_string = repair_json(raw_content)
                    parsed_json = json.loads(repaired_json_string)
                    err = "Warning: Original JSON was invalid and repaired with json_repair."
                except Exception as repair_exc:
                    err = f"Error: Failed after json_repair -> {repair_exc}. Raw response: '{raw_content[:100]}...'"
                    parsed_json = None
            if parsed_json and isinstance(parsed_json, dict):
                if "answer" in parsed_json:
                    extracted_answer = parsed_json["answer"]
                    err = None  # Parsing succeeded, clear previous error (if json_repair was successful)
                    if question_type == "single":
                        ans_val = extracted_answer.strip().upper()
                        if ans_val in {'A', 'B', 'C', 'D'}:
                            extracted_answer = ans_val
                        else:
                            err = (err + "; " if err else "") + f"Error: 'answer' value '{extracted_answer}' is not A, B, C, or D."
                            extracted_answer = "Error_Invalid_Option_Value" 
                    elif question_type == "multiple":
                        potential_answer_obj = extracted_answer
                        if isinstance(potential_answer_obj, list): 
                            valid_options_format = True
                            processed_selection: List[str] = []
                            for item_raw in potential_answer_obj:
                                if not isinstance(item_raw, str):
                                    valid_options_format = False
                                    err = (
                                        err + "; " if err else "") + f"Error: 'answer' list contains a non-string element '{item_raw}'."
                                    break
                                item = item_raw.strip().upper()
                                if not item and potential_answer_obj == ['']:  # handle {"answer": [""]} as {"answer": []}
                                    continue

                                if actual_option_labels:
                                    if item in actual_option_labels:
                                        processed_selection.append(item)
                                    else:
                                        valid_options_format = False
                                        err = (
                                            err + "; " if err else "") + f"Error: 'answer' list contains an invalid option '{item}'. Valid options: {actual_option_labels}. Full content: {potential_answer_obj}"
                                        break
                                else:  # Fallback if actual_option_labels couldn't be derived
                                    if len(item) == 1 and 'A' <= item <= 'Z':
                                        processed_selection.append(item)
                                    else:
                                        valid_options_format = False
                                        err = (
                                            err + "; " if err else "") + f"Error: 'answer' list contains an invalid format '{item}' (no option labels provided for validation)."
                                        break

                            if valid_options_format:
                                extracted_answer = sorted(list(set(processed_selection)))  # Standardize
                                # If err was just the warning from json_repair, it's fine.
                                # If err was set by validation, extracted_answer will remain "Error..."
                            else:
                                # extracted_answer remains "Error..."
                                pass

                        elif potential_answer_obj is None:  # "answer": null
                            err = (err + "; " if err else "") + "Error: The value of the 'answer' key is null; expected a list (e.g., [] for no answer)."
                        else:  # "answer" key exists but value is not a list or null
                            err = (
                                err + "; " if err else "") + f"Error: The value of the 'answer' key is not a list. Type: {type(potential_answer_obj)}"
                    elif question_type == "TF":
                        potential_answer_obj = extracted_answer
                        if isinstance(potential_answer_obj, str) and potential_answer_obj.strip():
                            ans_upper = potential_answer_obj.upper()
                            # Check if it starts with TRUE/FALSE or indicates insufficiency
                            if ans_upper.startswith("TRUE.") or \
                                    ans_upper.startswith("FALSE.") or \
                                    "INSUFFICIENT" in ans_upper or \
                                    "CANNOT DETERMINE" in ans_upper or \
                                    "NOT ENOUGH INFORMATION" in ans_upper:
                                extracted_answer = potential_answer_obj.strip()
                                # err might still hold the json_repair warning, which is fine.
                            else:
                                err = (
                                    err + "; " if err else "") + f"Error: 'answer' string format is not as expected (does not start with TRUE./FALSE. or indicate insufficiency). Content: '{potential_answer_obj[:100]}...'"
                        elif potential_answer_obj is None:
                            err = (err + "; " if err else "") + "Error: The value of the 'answer' key is null; expected a string."
                        elif isinstance(potential_answer_obj, str) and not potential_answer_obj.strip():
                            err = (err + "; " if err else "") + "Error: The value of the 'answer' key is an empty string."
                        else:
                            err = (
                                err + "; " if err else "") + f"Error: The value of the 'answer' key is not a string. Type: {type(potential_answer_obj)}"
                else:
                    # Keep the repair_exc error message (if any), otherwise set a new error message
                    err = err or "Error: JSON is missing the 'answer' key or 'answer' is not a string."
            elif not err:  # If parsed is empty or of wrong type, and no previous error (i.e., both parsing and repair failed)
                err = "Error: The response from LLM is not a valid JSON object."



    except Exception as e:
        err = f"Error: Unexpected exception during API interaction. {e}"
        print(
            f"Unexpected exception during API interaction Q: '{question[:50]}...' (Model: {llm_model}, Temp: {temperature}, Top_p:{top_p}): {e}.")
        if raw_content is None:
            raw_content = ""
        extracted_answer = "Error_Unexpected_Exception"
        

    if raw_content is None:
        raw_content = ""

    return {
        "raw_response": raw_content,
        "extracted_answer": extracted_answer,
        "error": err,
    }

def process_question_item(qobj: Dict[str, Any], llm_model: str, question_type: str, temperature: float, top_p: float) -> Optional[Dict[str, Any]]:
    
    llm_result = call_llm(qobj, llm_model, question_type, temperature, top_p)
    processed_record = dict(qobj)
    processed_record["model_evaluation_result"] = {
        "model_raw_response": llm_result["raw_response"],
        "model_answer": llm_result["extracted_answer"],
        "error": llm_result["error"],
    }
    return processed_record



