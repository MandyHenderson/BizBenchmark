import os
import json
import logging
import re # For sanitizing model names for paths
import copy # Import copy module
import asyncio # For asynchronous operations
from collections import defaultdict # For grouping questions
from typing import Any, Dict, List, Tuple, Optional

from dataloader.loader import load_all_questions
from prompt.prompt import format_prompt
from model.base_model import BaseModel # Import BaseModel
from model.api_model import APIModel
# from model.local_model import LocalModel # Moved import
# from evaluation.evaluate import evaluate_results # Placeholder evaluation will be inline

# --- Configuration ---
# Use the specific data directory provided by the user
DATA_DIR = r""
# API_KEY_ENV_VAR = "DEEPSEEK_API_KEY" # Removed, API key will be hardcoded

# Model Selection
USE_API_MODEL = True  # Set to False to use LocalModel
# Define the local model name/path if USE_API_MODEL is False
LOCAL_MODEL_NAME_OR_PATH = "sshleifer/tiny-gpt2" # Example, replace with your desired local model
# API Model parameters (can be overridden in APIModel constructor if needed)
API_MODEL_NAME = "deepseek-chat"
API_BASE_URL = "https://api.deepseek.com"

BASE_OUTPUT_DIR = "eval_outputs" # Base directory for all evaluation outputs

QUESTION_LIMIT_FOR_TESTING = None # Set to a number (e.g., 5) for quick testing, None for all questions
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Please follow the user instructions precisely and provide your answer in the specified JSON format."
CONCURRENT_API_CALL_LIMIT = 10 # Configurable limit for concurrent API calls

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sanitize_model_name(model_name: str) -> str:
    """Sanitizes the model name to be used as a valid directory/file name."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', model_name)

# def normalize_answer_for_comparison(answer: Any) -> Any:
#     """Normalizes an answer for comparison. Converts to lower case if string."""
#     if isinstance(answer, str):
#         return answer.strip().lower()
#     if isinstance(answer, list):
#         # Ensure all items are strings before sorting and joining for comparison
#         return sorted([str(item).strip().lower() for item in answer])
#     return answer

# def evaluate_single_response(
#     model_output_json: Optional[Dict[str, Any]],
#     ground_truth_answer: Any,
#     question_type: str,
#     parse_error: Optional[str]
# ) -> Tuple[bool, str]:
#     """
#     Evaluates a single model response against the ground truth.
#     This is a placeholder and needs to be adapted based on actual ground truth format and evaluation criteria.
#     """
#     if parse_error:
#         return False, f"JSON_PARSE_ERROR: {parse_error}"
#     if model_output_json is None:
#         return False, "MODEL_OUTPUT_EMPTY_OR_NONE"

#     model_answer_field = model_output_json.get("answer")
#     if question_type == "Fill-in-the Blank":
#         model_answer_field = model_output_json.get("answers")

#     if model_answer_field is None:
#         if len(model_output_json) == 1:
#             first_key = list(model_output_json.keys())[0]
#             logging.warning(f"Missing 'answer' or 'answers' key directly. Trying first key '{first_key}' as potential wrapper.")
#             model_answer_field = model_output_json.get(first_key)
#             if isinstance(model_answer_field, dict): 
#                 model_answer_field = model_answer_field.get("answer") 
#                 if question_type == "Fill-in-the Blank" and not model_answer_field:
#                     model_answer_field = model_output_json.get(first_key).get("answers")

#     if model_answer_field is None:
#         return False, f"ANSWER_FIELD_MISSING_IN_MODEL_JSON. Keys found: {list(model_output_json.keys())}"
    
#     norm_model_answer = normalize_answer_for_comparison(model_answer_field)
#     norm_ground_truth = normalize_answer_for_comparison(ground_truth_answer)

#     if question_type == "TF":
#         if isinstance(norm_model_answer, str) and isinstance(norm_ground_truth, str):
#             model_decision = norm_model_answer.split('.')[0]
#             is_correct = model_decision == norm_ground_truth
#             return is_correct, f"Compared model decision '{model_decision}' with ground truth '{norm_ground_truth}'"
#         else:
#             return False, f"Type mismatch for TF comparison (model: {type(norm_model_answer)}, gt: {type(norm_ground_truth)})"
            
#     elif question_type in ["Single Choice", "General QA", "Numerical QA", "Proof", "Table QA"]:
#         is_correct = norm_model_answer == norm_ground_truth
#         return is_correct, f"Compared normalized answers (model: {str(norm_model_answer)[:50]}, gt: {str(norm_ground_truth)[:50]})"
        
#     elif question_type == "Multiple Choice":
#         if isinstance(norm_model_answer, list) and isinstance(norm_ground_truth, list):
#             is_correct = norm_model_answer == norm_ground_truth
#             return is_correct, "Compared normalized lists for multiple choice."
#         else:
#             return False, f"Type mismatch for Multiple Choice (model: {type(norm_model_answer)}, gt: {type(norm_ground_truth)})"

#     elif question_type == "Fill-in-the Blank":
#         if isinstance(norm_model_answer, list) and isinstance(norm_ground_truth, list):
#             is_correct = norm_model_answer == norm_ground_truth
#             return is_correct, "Compared normalized lists for fill-in-the-blank."
#         elif isinstance(norm_ground_truth, str) and isinstance(norm_model_answer, list) and len(norm_model_answer) == 1:
#             is_correct = norm_model_answer[0] == norm_ground_truth
#             return is_correct, "Compared single GT string with model's list of one for fill-in-the-blank."
#         return False, f"Type mismatch or structure mismatch for Fill-in-the-Blank (model: {type(norm_model_answer)}, gt: {type(norm_ground_truth)})"
            
#     else:
#         return False, f"UNKNOWN_QUESTION_TYPE_FOR_EVALUATION: {question_type}"

#     return False, "EVALUATION_LOGIC_NOT_FULLY_IMPLEMENTED_FOR_TYPE"

async def process_single_question(
    original_question_data: Dict[str, Any],
    model: BaseModel,
    q_idx: int, 
    total_q_for_file: int,
    relative_file_path: str
) -> Dict[str, Any]:
    
    result_item = copy.deepcopy(original_question_data)
    q_id = result_item.get("qid", f"unknown_qid_for_{os.path.basename(relative_file_path)}_{q_idx}")
    q_type = result_item.get("question_source_type", "UnknownType")

    logging.info(f"Processing Q {q_idx}/{total_q_for_file} from {relative_file_path} (ID: {q_id}, Type: {q_type})")

    user_prompt = format_prompt(original_question_data) 

    model_eval_result = {
        "model_raw_response": None,
        "model_answer": None,
        "json_parse_error": None
    }

    infer_kwargs: Dict[str, Any] = {
        "expect_json_object": True,
        "system_prompt": DEFAULT_SYSTEM_PROMPT
    }

    try:
        # Ensure model.infer is awaited if it's an async method (which APIModel.infer now is)
        raw_model_response = await model.infer(user_prompt=user_prompt, **infer_kwargs) # type: ignore
        model_eval_result["model_raw_response"] = raw_model_response

        if raw_model_response:
            try:
                match = re.search(r"```json\\n(.*?)\\n```", raw_model_response, re.DOTALL)
                json_str_to_parse = match.group(1) if match else raw_model_response
                
                model_output_json = json.loads(json_str_to_parse)
                
                if isinstance(model_output_json, dict):
                    if q_type == "Fill-in-the Blank" and "answers" in model_output_json:
                        model_eval_result["model_answer"] = model_output_json.get("answers")
                    elif "answer" in model_output_json:
                        model_eval_result["model_answer"] = model_output_json.get("answer")
                    else:
                        logging.warning(f"QID {q_id}: Valid JSON, but 'answer'/'answers' key missing. Storing full JSON. Keys: {list(model_output_json.keys())}")
                        model_eval_result["model_answer"] = model_output_json
                else:
                    logging.warning(f"QID {q_id}: Parsed JSON is not a dict. Storing as is. Type: {type(model_output_json)}")
                    model_eval_result["model_answer"] = model_output_json

            except json.JSONDecodeError as jde:
                model_eval_result["json_parse_error"] = str(jde)
                logging.warning(f"QID {q_id}: JSONDecodeError. Response: {raw_model_response[:300]}... Error: {jde}")
        else:
            model_eval_result["json_parse_error"] = "MODEL_RETURNED_EMPTY_RESPONSE"
            logging.warning(f"QID {q_id}: Empty response from model.")
    
    except Exception as e:
        logging.error(f"QID {q_id}: Error during model inference: {e}", exc_info=True)
        model_eval_result["json_parse_error"] = f"INFERENCE_ERROR: {str(e)}"
    
    result_item["model_evaluation_result"] = model_eval_result
    return result_item

async def run_evaluation(data_dir: str, question_limit: Optional[int] = None):
    logging.info(f"Starting evaluation run. USE_API_MODEL: {USE_API_MODEL}, Concurrent API Calls: {CONCURRENT_API_CALL_LIMIT}")

    model: Optional[BaseModel] = None
    model_identifier: str = "unknown_model"

    if USE_API_MODEL:
        hardcoded_api_key = ""
        if not hardcoded_api_key: 
            logging.error("API key is missing.")
            return
        try:
            model = APIModel(api_key=hardcoded_api_key, model_name=API_MODEL_NAME, base_url=API_BASE_URL)
            model_identifier = sanitize_model_name(API_MODEL_NAME)
            logging.info(f"APIModel initialized: {API_MODEL_NAME}")
        except Exception as e:
            logging.error(f"Failed to initialize APIModel: {e}")
            return
    else:
        from model.local_model import LocalModel 
        try:
            model = LocalModel(model_name_or_path=LOCAL_MODEL_NAME_OR_PATH)
            if hasattr(model, 'model') and model.model is not None: 
                model_identifier = sanitize_model_name(LOCAL_MODEL_NAME_OR_PATH)
                logging.info(f"LocalModel initialized: {LOCAL_MODEL_NAME_OR_PATH}")
            else:
                logging.error(f"LocalModel '{LOCAL_MODEL_NAME_OR_PATH}' failed to load. Cannot proceed.")
                return
        except Exception as e:
            logging.error(f"Failed to initialize LocalModel: {e}")
            return

    if model is None: 
        logging.error("Model could not be initialized.")
        return

    base_model_output_dir = os.path.join(BASE_OUTPUT_DIR, model_identifier)
    os.makedirs(base_model_output_dir, exist_ok=True)
    logging.info(f"Base output directory for this model: {base_model_output_dir}")

    logging.info(f"Loading questions from data directory: {data_dir}")
    all_questions_flat, file_counts = load_all_questions(data_dir)
    if not all_questions_flat:
        logging.error("No questions loaded.")
        return
    
    logging.info(f"Successfully loaded {len(all_questions_flat)} questions from {len(file_counts)} files.")

    questions_by_source_file = defaultdict(list)
    for q in all_questions_flat:
        relative_path = q.get('original_file_relative_path')
        if relative_path:
            questions_by_source_file[relative_path].append(q)
        else:
            logging.warning(f"Question {q.get('qid', 'Unknown QID')} is missing 'original_file_relative_path'. Skipping.")

    if question_limit is not None and question_limit > 0:
        logging.warning(f"QUESTION_LIMIT_FOR_TESTING ({question_limit}) is active. This will only process a subset of files/questions.")
        # This simple truncation might not be ideal with per-file processing. 
        # For true limited run, one might pick first N files or first N questions from first file.
        # For now, it limits total questions processed across files by processing files until limit is hit.
        pass # Limit will be handled inside file loop implicitly by total_processed_count

    total_questions_processed_across_all_files = 0

    for relative_file_path, questions_in_file in questions_by_source_file.items():
        if question_limit and total_questions_processed_across_all_files >= question_limit:
            logging.info(f"Global question limit ({question_limit}) reached. Stopping further file processing.")
            break

        logging.info(f"--- Processing source file: {relative_file_path} ({len(questions_in_file)} questions) ---")

        # Create specific output directory for this source file
        path_parts = os.path.normpath(relative_file_path).split(os.sep)
        file_specific_subdir_name = os.path.splitext(path_parts[-1])[0] # Filename without extension
        if len(path_parts) > 1:
            intermediate_dirs = os.path.join(*path_parts[:-1])
            output_dir_for_this_source_file = os.path.join(base_model_output_dir, intermediate_dirs, file_specific_subdir_name)
        else:
            output_dir_for_this_source_file = os.path.join(base_model_output_dir, file_specific_subdir_name)
        
        os.makedirs(output_dir_for_this_source_file, exist_ok=True)

        output_jsonl_file_path = os.path.join(output_dir_for_this_source_file, f"results_{model_identifier}.jsonl")
        output_final_json_file_path = os.path.join(output_dir_for_this_source_file, f"final_results_{model_identifier}.json")
        logging.info(f"Output for this file will be in: {output_dir_for_this_source_file}")

        processed_items_for_this_file = []
        with open(output_jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
            tasks_for_current_file = []
            for q_idx, original_q_data in enumerate(questions_in_file):
                if question_limit and total_questions_processed_across_all_files >= question_limit:
                    break 
                tasks_for_current_file.append(process_single_question(original_q_data, model, q_idx + 1, len(questions_in_file), relative_file_path))
                total_questions_processed_across_all_files +=1
            
            # Process tasks in batches to respect CONCURRENT_API_CALL_LIMIT
            for i in range(0, len(tasks_for_current_file), CONCURRENT_API_CALL_LIMIT):
                batch_tasks = tasks_for_current_file[i:i+CONCURRENT_API_CALL_LIMIT]
                logging.info(f"Running batch of {len(batch_tasks)} API calls for {relative_file_path}...")
                results_batch = await asyncio.gather(*batch_tasks)
                for result_item in results_batch:
                    if result_item: # Ensure no None results if a task failed catastrophically before returning dict
                        processed_items_for_this_file.append(result_item)
                        try:
                            jsonl_file.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                            jsonl_file.flush()
                        except Exception as e:
                            logging.error(f"Error writing QID {result_item.get('qid')} to jsonl for {relative_file_path}: {e}")
                if question_limit and total_questions_processed_across_all_files >= question_limit and i+CONCURRENT_API_CALL_LIMIT < len(tasks_for_current_file):
                    logging.info(f"Global question limit ({question_limit}) reached mid-file {relative_file_path}. Remaining questions in this file will be skipped.")
                    break
        
        # Convert .jsonl to final .json for this specific file
        logging.info(f"Converting {output_jsonl_file_path} to {output_final_json_file_path}")
        if not processed_items_for_this_file: # If list is empty (e.g. question_limit hit before any processing for this file)
            logging.info(f"No items processed for {relative_file_path} to save to final JSON. Skipping conversion.")
        else:
            try:
                with open(output_final_json_file_path, 'w', encoding='utf-8') as f_final_json:
                    json.dump(processed_items_for_this_file, f_final_json, indent=4, ensure_ascii=False)
                logging.info(f"Final results for {relative_file_path} saved to: {output_final_json_file_path}")
            except Exception as e:
                logging.error(f"An error occurred during .json conversion for {relative_file_path}: {e}")
            
    logging.info("Evaluation run completed for all source files.")

if __name__ == "__main__":
    logging.info(f"Attempting to run evaluation with DATA_DIR: {DATA_DIR}")
    asyncio.run(run_evaluation(data_dir=DATA_DIR, question_limit=QUESTION_LIMIT_FOR_TESTING))
