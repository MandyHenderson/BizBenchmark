"""model.py

Batch driver that feeds question JSON files to an LLM, captures its answers,
and writes evaluation artefacts (.jsonl files and a merged .json file).

Usage workflow
==============
1. Arrange the data directory as follows::

        <input_dir>/
            single/          # question_type
                gpt-4o/      # model_name
                    batch.json  # one or more batches

2. Call `process_directory` for every combination of `question_type`
   and `model_name` to evaluate. The function handles resume logic,
   thread-pool parallelism, and fault tolerance.

Business logic lives in three functions so each can be unit-tested:

- process_directory     – orchestrates directory walking and writes output
- call_llm              – builds prompts and calls the OpenAI-compatible API
- process_question_item – attaches the LLM result back to the original item

Docstrings and inline comments are in English so the code can be shared
easily with international collaborators.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

from dataloader.dataloader import load_questions
from utils.prompt import *  # noqa: F401, provides *single_prompt*, *multiple_prompt*, etc.
from utils.utils import *   # noqa: F401, needed for *split_options*, *UpstreamSaturationError*, etc.
from json_repair import repair_json

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MAX_WORKERS = 1  # ThreadPool size. Increase cautiously; many providers rate‑limit.
MAX_LLM_RETRIES = 3  # Not used in this snippet but reserved for exponential‑backoff logic.
LLM_RETRY_DELAY = 10  # Seconds between retries when 503 / rate‑limit occurs.
MAX_TOKENS_FOR_JSON = 512  # Max tokens requested from the LLM API.
UPSTREAM_SATURATION_KEYWORD = "Saturation Detected"  # Vendor‑specific flag to halt batch.

# !!! Replace with **your** key, or load from environment / vault in production.
API_KEY = "sk-"
BASE_URL = "https://api.deepseek.com/v1"  # Might need adjustment for different hosts.

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# =============================================================================
# Orchestration helpers
# =============================================================================

def process_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    question_type: str,
    model_name: str,
    temperature: float,
    top_p: float,
) -> None:
    """Process one directory containing question JSON files.

    Steps performed:
    1. Iterate over every .json file directly under *input_dir*.
    2. Split each file into previously processed and pending questions using
       `load_questions`.
    3. Submit pending questions to a thread pool (size `MAX_WORKERS`) and
       process them with `process_question_item`; track progress with *tqdm*.
    4. Append each finished record to an .jsonl file immediately, then write a
       deduplicated .json file when all tasks finish.

    Args:
        input_dir: Directory holding question JSON files.
        output_dir: Destination root; a sub-folder named *evaluation* is
            created for output artefacts.
        question_type: Controls the prompt template (for example, "single"
            or "multiple").
        model_name: Identifier for the LLM model.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling parameter.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Basic run metadata for operator visibility.
    print(f"output_dir: {output_dir}")
    print(f"model_name: {model_name}")
    print(f"temperature: {temperature}")
    print(f"top_p: {top_p}")

    # Each batch file will have an *evaluation* subfolder.
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Only process files at *first level*; skip nested dirs.
    json_files_to_process = [
        f for f in input_dir.glob("*.json") if f.is_file() and f.parent.resolve() == input_dir.resolve()
    ]

    for input_json_file_path in json_files_to_process:
        # -------------- Prepare per‑file paths & resume state -------------- #
        eval_jsonl_path = eval_dir / f"{input_json_file_path.stem}_eval.jsonl"
        eval_json_path = eval_dir / f"{input_json_file_path.stem}_eval.json"

        previously_processed_records, questions_to_process_now = load_questions(
            input_json_file_path, eval_jsonl_path
        )
        # Copy to avoid mutating the list returned by loader.
        all_records_for_this_file: List[Dict[str, Any]] = list(previously_processed_records)
        newly_processed_and_written_count = 0

        # ------------------- Process new questions via LLM ------------------ #
        if questions_to_process_now:
            print(f"    Processing {len(questions_to_process_now)} new or reprocessable question items…")

            progress_bar: Optional[tqdm] = None
            future_to_qobj: Dict[concurrent.futures.Future, Dict[str, Any]] = {}

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    # Submit every question to thread pool.
                    for qobj_to_submit in questions_to_process_now:
                        future = executor.submit(
                            process_question_item,
                            qobj_to_submit,
                            model_name,
                            question_type,
                            temperature,
                            top_p,
                        )
                        future_to_qobj[future] = qobj_to_submit

                    # tqdm can iterate over *as_completed* for dynamic progress.
                    pbar_desc = (
                        f"LLM({model_name}|T:{temperature}|Top_p:{top_p}) "
                        f"{input_json_file_path.name[:15]}"
                    )
                    progress_bar = tqdm(
                        concurrent.futures.as_completed(future_to_qobj),
                        total=len(future_to_qobj),
                        desc=pbar_desc,
                        unit="q",
                        ncols=120,
                    )

                    # Collect results as each future resolves.
                    for future in progress_bar:
                        original_qobj_for_future = future_to_qobj[future]
                        qid_for_log = original_qobj_for_future.get("qid", "NO_QID_IN_ORIGINAL_FOR_FUTURE")

                        try:
                            result_record = future.result()
                            if result_record:
                                all_records_for_this_file.append(result_record)
                                # Append line‑by‑line to preserve progress.
                                with eval_jsonl_path.open("a", encoding="utf-8") as f_jsonl:
                                    f_jsonl.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                                newly_processed_and_written_count += 1

                        except UpstreamSaturationError as use:
                            # Hard exit: provider signalled *global* saturation.
                            if progress_bar:
                                progress_bar.write(
                                    f"\n[FATAL ERROR] Upstream saturation (QID: {qid_for_log}, Model: {model_name}, "
                                    f"Temp: {temperature}, Top_p: {top_p}): {use}."
                                )
                            executor.shutdown(wait=True, cancel_futures=True)
                            if progress_bar:
                                progress_bar.close()
                            print("[INFO] Terminating due to upstream saturation.")
                            sys.exit(1)

                        except KeyboardInterrupt:
                            # Respect Ctrl‑C quickly.
                            if progress_bar:
                                progress_bar.write("\n[INFO] User interrupted during task result waiting.")
                            executor.shutdown(wait=True, cancel_futures=True)
                            if progress_bar:
                                progress_bar.close()
                            print("[INFO] Processing stopped. Program will exit.")
                            sys.exit(0)

                        except Exception as exc_task:
                            # Catch‑all: log and insert placeholder so downstream
                            # scripts don't crash on missing entries.
                            if progress_bar:
                                progress_bar.write(
                                    f"[ERROR] QID '{qid_for_log}' (Model: {model_name}, Temp: {temperature}, "
                                    f"Top_p: {top_p}) encountered an unexpected framework exception: {exc_task}."
                                )
                            qid_from_original = original_qobj_for_future.get("qid")
                            if isinstance(qid_from_original, str) and qid_from_original.strip():
                                error_placeholder = {
                                    **original_qobj_for_future,
                                    "model_evaluation_result": {
                                        "model_raw_response": "",
                                        "model_answer": "Error_Framework_Exception",
                                        "error": f"Task framework exception: {str(exc_task)}",
                                    },
                                }
                                all_records_for_this_file.append(error_placeholder)
                                with eval_jsonl_path.open("a", encoding="utf-8") as f_jsonl:
                                    f_jsonl.write(json.dumps(error_placeholder, ensure_ascii=False) + "\n")
                                newly_processed_and_written_count += 1
                            # else: cannot record placeholder without a valid QID.

            # ------------------- Outer try/except around pool ------------------ #
            except KeyboardInterrupt:
                print("\n[INFO] User interrupted during task submission/setup.")
                if progress_bar and not progress_bar.disable:
                    progress_bar.close()
                print("[INFO] Processing stopped. User interrupt will cause program to exit.")
                sys.exit(0)
            except Exception as pool_exc:
                print(
                    f"[ERROR] Unexpected error during thread pool execution (Model: {model_name}, "
                    f"Temp: {temperature}, Top_p: {top_p}): {pool_exc}"
                )
            finally:
                if progress_bar and not progress_bar.disable:
                    progress_bar.close()

            # Report batch‑level summary.
            if newly_processed_and_written_count > 0:
                print(
                    f"    LLM processing of new items completed. "
                    f"{newly_processed_and_written_count} results appended to {eval_jsonl_path.name}"
                )
            elif questions_to_process_now:
                print(
                    f"    LLM processing completed, but no items successfully produced results in "
                    f"{eval_jsonl_path.name} (possible API errors, skips, or data issues)."
                )

        # ------------------------- No new work to do ------------------------- #
        elif not previously_processed_records:
            print(f"    [INFO] {input_json_file_path.name} has no new items to process and no previous results found.")
        else:
            print(
                f"    [INFO] All items in {input_json_file_path.name} have been processed and loaded from "
                f"{eval_jsonl_path.name}."
            )

        # ----------------------- Write final merged JSON ---------------------- #
        if not all_records_for_this_file:
            print(f"    [INFO] No records (new or old) to summarize or save for {input_json_file_path.name}.")
            continue

        # Deduplicate by QID; fall back to synthetic key if missing.
        final_unique_records_map: Dict[str, Dict[str, Any]] = {}
        temp_id_counter = 0
        for rec in all_records_for_this_file:
            qid = rec.get("qid")
            if isinstance(qid, str) and qid.strip():
                final_unique_records_map[qid] = rec
            else:
                temp_key = f"__temp_no_qid_at_final_agg_{temp_id_counter}__"
                final_unique_records_map[temp_key] = rec
                temp_id_counter += 1

        final_records_list = list(final_unique_records_map.values())

        try:
            eval_json_path.write_text(
                json.dumps(final_records_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"    Merged evaluation data ({len(final_records_list)} unique/final records) "
                f"has been saved to {eval_json_path.name}"
            )
        except Exception as e:
            print(f"    [Error] Failed to write final merged JSON to {eval_json_path.name}: {e}")


# =============================================================================
# Low‑level helpers
# =============================================================================

def call_llm(
    qobj: Dict[str, Any],
    llm_model: str,
    question_type: str,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    """Call the LLM with one question and return its parsed answer.

    The function builds a system + user message pair, sends it via the global
    `client`, and enforces that the response is a JSON object. It extracts the
    `answer` field, performs type-specific validation, and returns:

        - raw_response: the raw text returned by the model (or empty string)
        - extracted_answer: the parsed `answer` value, or an error marker
        - error: None if parsing succeeded, otherwise an error message

    Validation rules by `question_type`:
        • single   – answer must be A, B, C, or D
        • multiple – answer must be a list of uppercase option labels
        • tf       – answer must start with TRUE./FALSE. or state insufficiency
    """
    if not client:
        raise ConnectionError("LLM client is not initialized. Please check your API key and base URL.")

    # ------------------------- Prompt assembly ------------------------- #
    user_prompt_parts: List[str] = [f"Question: {str(qobj.get('question')).strip()}"]

    # Template‑specific additions.
    if question_type == "single":
        # -------------------------- Single choice ------------------------- #
        raw_q_original = str(qobj.get("question")).strip()
        raw_q_for_llm = raw_q_original  # Might change if options are parsed.
        qid = qobj.get("qid") or f"NO_QID_INPUT_{raw_q_original[:20]}"

        # Background/context in dataset.
        context_str_bg = qobj.get("background_text", "")
        if isinstance(context_str_bg, str) and context_str_bg.strip():
            user_prompt_parts.append(f"\nContext:\n{context_str_bg.strip()}")

        # Either from explicit field, or parsed from question stem.
        options_list_input: Any = qobj.get("options")
        opts_list_processed: List[str] = []
        if isinstance(options_list_input, list) and options_list_input:
            opts_list_processed = [str(opt).strip() for opt in options_list_input if str(opt).strip()]
        else:
            # Fallback: attempt to split from the question text itself.
            processed_q_stem, parsed_opts = split_options(raw_q_original)
            if parsed_opts:
                raw_q_for_llm = processed_q_stem
                opts_list_processed = parsed_opts

        if opts_list_processed:
            user_prompt_parts.append("Options:\n" + "\n".join(opts_list_processed))

    elif question_type == "multiple":
        # ------------------------- Multiple choice ------------------------ #
        raw_options = qobj.get("options", [])
        if isinstance(raw_options, list) and all(isinstance(opt, str) for opt in raw_options):
            if raw_options:
                user_prompt_parts.append("Options:\n" + "\n".join(raw_options))

    # Rich media additions (optional fields) ------------------------------- #
    table_html = qobj.get("table_html")
    if table_html and str(table_html).strip():
        user_prompt_parts.append(f"\nTable (HTML):\n```html\n{str(table_html).strip()}\n```")

    formula_context = qobj.get("formula_context")
    if formula_context and str(formula_context).strip():
        user_prompt_parts.append(f"\nFormula Context:\n{str(formula_context).strip()}")

    question_context = qobj.get("question_context")
    if question_context and str(question_context).strip():
        user_prompt_parts.append(f"\nAdditional Context:\n{str(question_context).strip()}")

    user_prompt = "\n\n".join(user_prompt_parts)

    # Choose prompt skeleton.
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
    else:
        raise ValueError(f"Unsupported question_type: {question_type}")

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_content: Optional[str] = None
    extracted_answer: str | List[str] = "Error_LLM_Logic_Failed"
    err: Optional[str] = None
    parsed_json: Optional[Dict[str, Any]] = None

    # ------------------------- API invocation -------------------------- #
    try:
        rsp = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=MAX_TOKENS_FOR_JSON,
            response_format={"type": "json_object"},
            stream=False,
            timeout=60.0,
        )
        # The SDK guarantees that *choices* exists, but we double‑check.
        raw_content = rsp.choices[0].message.content if rsp.choices and rsp.choices[0].message else ""

        # Provider‑level back‑pressure token.
        if raw_content and UPSTREAM_SATURATION_KEYWORD in raw_content:
            raise UpstreamSaturationError(
                f"Upstream saturation detected in response: {raw_content[:200]}…"
            )

        # --------------------- JSON parsing & repair -------------------- #
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
                    err = f"Error: Failed after json_repair -> {repair_exc}. Raw response: '{raw_content[:100]}…'"
                    parsed_json = None

            # ---------------- Validate & normalise answer field ---------------- #
            if parsed_json and isinstance(parsed_json, dict):
                if "answer" in parsed_json:
                    extracted_answer = parsed_json["answer"]
                    err = None  # Clear previous warnings if answer looks okay; further validation below.

                    # ----------- Type‑specific validation ----------- #
                    if question_type == "single":
                        ans_val = str(extracted_answer).strip().upper()
                        if ans_val in {"A", "B", "C", "D"}:
                            extracted_answer = ans_val
                        else:
                            err = (
                                (err + "; " if err else "") +
                                f"Error: 'answer' value '{extracted_answer}' is not A, B, C, or D."
                            )
                            extracted_answer = "Error_Invalid_Option_Value"

                    elif question_type == "multiple":
                        # Guard against wrong types.
                        if isinstance(extracted_answer, list):
                            processed_selection = [str(item).strip().upper() for item in extracted_answer if str(item).strip()]
                            extracted_answer = sorted(list(set(processed_selection)))
                        else:
                            err = (
                                (err + "; " if err else "") +
                                "Error: 'answer' key must be a list for 'multiple' question_type."
                            )
                            extracted_answer = "Error_Invalid_List"

                    elif question_type == "tf":  # True/False style.
                        if isinstance(extracted_answer, str):
                            ans_upper = extracted_answer.upper()
                            if not (
                                ans_upper.startswith("TRUE.") or ans_upper.startswith("FALSE.") or
                                "INSUFFICIENT" in ans_upper or "CANNOT DETERMINE" in ans_upper or
                                "NOT ENOUGH INFORMATION" in ans_upper
                            ):
                                err = (
                                    (err + "; " if err else "") +
                                    "Error: 'answer' string format is not as expected for TF question_type."
                                )
                        else:
                            err = (
                                (err + "; " if err else "") +
                                "Error: 'answer' key must be a string for TF question_type."
                            )
                else:
                    err = err or "Error: JSON is missing the 'answer' key."
            elif not err:
                err = "Error: The response from LLM is not a valid JSON object."

    except Exception as e:
        # Catch *any* exception so that the pipeline can continue.
        err = f"Error: Unexpected exception during API interaction. {e}"
        print(
            f"Unexpected exception during API interaction Q: '{str(qobj.get('question'))[:50]}…' (Model: {llm_model}, Temp: {temperature}, Top_p:{top_p}): {e}."
        )
        raw_content = raw_content or ""
        extracted_answer = "Error_Unexpected_Exception"

    return {
        "raw_response": raw_content or "",
        "extracted_answer": extracted_answer,
        "error": err,
    }


def process_question_item(
    qobj: Dict[str, Any],
    llm_model: str,
    question_type: str,
    temperature: float,
    top_p: float,
) -> Optional[Dict[str, Any]]:
    """Return a copy of the question object augmented with the LLM result.

    The new key `model_evaluation_result` contains:

        {
            "model_raw_response": <raw text from the LLM>,
            "model_answer":       <parsed answer or error marker>,
            "error":              <None or error message>
        }
    """

    llm_result = call_llm(qobj, llm_model, question_type, temperature, top_p)
    processed_record = dict(qobj)  # Shallow copy to avoid mutating caller's dict.
    processed_record["model_evaluation_result"] = {
        "model_raw_response": llm_result["raw_response"],
        "model_answer": llm_result["extracted_answer"],
        "error": llm_result["error"],
    }
    return processed_record
