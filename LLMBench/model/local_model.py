"""Local model interface for BizBenchmark evaluation framework.

This module provides a complete interface for running local language models
using DeepSpeed for distributed inference. It supports both single-GPU and
multi-GPU deployments with automatic batch processing for improved performance.

The module follows a two-phase architecture:
1. One-time initialization by main.py at program startup
2. Business logic functions for inference and evaluation

Key Features:
    - DeepSpeed-based distributed inference
    - Automatic batch processing for GPU efficiency
    - OpenAI-compatible API interface
    - Resume functionality for long-running evaluations
    - Comprehensive error handling and retry logic

Example:
    # Initialize model (called by main.py)
    initialize_local_model()
    
    # Single inference
    result = simple_inference("What is the capital of France?")
    
    # Batch processing
    task_configs = [{"domain": "ECON", "question_type": "single", ...}]
    success, failed = run_batch_local(task_configs)

Note:
    This module requires DeepSpeed and appropriate GPU resources.
    Model initialization must occur at the top level before any inference.
"""

import os
import time
import torch
import json
import sys
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

# Model configuration
MODEL_DIR = "/home/bld/data/data2/Yichun_Lu/fino1/deepseek-math-7b-instruct"
DTYPE = torch.float16
TP_SIZE = 4
MAX_NEW_TOKENS = 8192

# Global variables - store initialized model
_engine = None
_tokenizer = None
_local_rank = 0
_is_initialized = False  # Initialization flag

# Import tools
from dataloader.dataloader import load_questions
from utils.prompt import *
from utils.utils import *
from json_repair import repair_json

# Constants
MAX_WORKERS = 1
BATCH_SIZE = 14  # Batch inference size, can be adjusted based on GPU memory
UPSTREAM_SATURATION_KEYWORD = "Saturation Detected"


class LocalModelError(Exception):
    """Custom exception for local model operations."""
    pass


def initialize_local_model():
    """Initializes local model with DeepSpeed for distributed inference.
    
    This function performs one-time model initialization and must be called
    by the main program at the top level before any inference operations.
    It handles both single-GPU and distributed multi-GPU setups.
    
    The initialization process:
    1. Sets up distributed environment if multiple GPUs detected
    2. Loads tokenizer with appropriate padding configuration
    3. Loads model with specified precision
    4. Initializes DeepSpeed inference engine for optimization
    
    Raises:
        RuntimeError: If model initialization fails.
        
    Note:
        This function is idempotent - calling it multiple times is safe.
        Only rank 0 process prints initialization messages in distributed mode.
    """
    global _engine, _tokenizer, _local_rank, _is_initialized
    if _is_initialized:
        if int(os.getenv("RANK", "0")) == 0:
            print("Model already initialized, skipping.")
        return

    # 1) Distributed initialization
    if "LOCAL_RANK" in os.environ:
        import deepspeed
        _local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if _local_rank == 0:
            print("🚀 Distributed mode: Initializing DeepSpeed...")
        deepspeed.init_distributed(dist_backend="nccl")
        torch.cuda.set_device(_local_rank)

        if _local_rank == 0:
            print(f"[Rank 0] Loading tokenizer and model...")
    else:
        print("🔧 Single GPU mode...")

    # 2) Load tokenizer and model
    from transformers import AutoTokenizer, AutoModelForCausalLM

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    _tokenizer.padding_side = 'left'
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=DTYPE)
    
    # 3) DeepSpeed inference engine
    if "LOCAL_RANK" in os.environ:
        max_total_tokens = 5120
        ds_cfg = {
            "tensor_parallel": {"tp_size": TP_SIZE},
            "replace_with_kernel_inject": False,
            "max_out_tokens": max_total_tokens
        }
        _engine = deepspeed.init_inference(model, config=ds_cfg, dtype=DTYPE)
        
        if _local_rank == 0:
            print(f"[Rank 0] DeepSpeed engine ready! ✅")
    else:
        _engine = model.cuda()
        print("✅ Single GPU model loading completed")
    
    _is_initialized = True


def get_model():
    """Gets initialized model instance with validation.
    
    Returns:
        Tuple[torch.nn.Module, transformers.AutoTokenizer]: Model engine and tokenizer.
        
    Raises:
        LocalModelError: If model is not initialized.
    """
    if not _is_initialized:
        raise LocalModelError("Model not initialized. Please ensure initialize_local_model() is called at the top level in main.py.")
    return _engine, _tokenizer


def simple_inference(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Performs simple single-prompt inference.
    
    Args:
        prompt (str): Input text prompt for generation.
        max_tokens (int): Maximum tokens to generate.
        temperature (float): Sampling temperature (0.0 = deterministic).
        
    Returns:
        str: Generated response text.
    """
    results = simple_inference_batch([prompt], max_tokens, temperature)
    return results[0] if results else ""


def simple_inference_batch(prompts: List[str], max_tokens: int = 512, temperature: float = 0.1) -> List[str]:
    """Performs batch inference for multiple prompts to improve GPU utilization.
    
    This function processes multiple prompts simultaneously, which is more
    efficient than sequential processing for GPU utilization.
    
    Args:
        prompts (List[str]): List of input prompts for generation.
        max_tokens (int): Maximum tokens to generate per prompt.
        temperature (float): Sampling temperature (0.0 = deterministic).
        
    Returns:
        List[str]: List of generated responses corresponding to input prompts.
        
    Note:
        In distributed mode, only rank 0 returns actual results.
        Other ranks return empty strings to maintain list length consistency.
    """
    _engine, _tokenizer = get_model()
    
    if not prompts:
        return []
    
    # tokenize - batch processing
    inputs = _tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    if "LOCAL_RANK" in os.environ:
        inputs = inputs.to(_engine.module.device)
    else:
        inputs = inputs.to(_engine.device)
    
    # Generate
    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": min(max_tokens, MAX_NEW_TOKENS),
            "do_sample": temperature > 0.0,
            "pad_token_id": _tokenizer.eos_token_id,
        }
        
        if temperature > 0.0:
            generation_kwargs.update({
                "temperature": temperature,
                "top_p": 0.9,
            })
        
        # Distributed synchronization
        if "LOCAL_RANK" in os.environ:
            torch.cuda.synchronize()
            import deepspeed
            deepspeed.comm.barrier()
        
        outputs = _engine.generate(**inputs, **generation_kwargs)
        
        if "LOCAL_RANK" in os.environ:
            torch.cuda.synchronize()
            import deepspeed
            deepspeed.comm.barrier()
    
    # Decode (only return on rank 0)
    current_rank = int(os.getenv("RANK", "0"))
    if current_rank == 0:
        decoded_texts = _tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract answer for each prompt
        answers = []
        for i, (original_prompt, decoded) in enumerate(zip(prompts, decoded_texts)):
            if decoded.startswith(original_prompt):
                answer = decoded[len(original_prompt):].strip()
            else:
                answer = decoded.strip()
            answers.append(answer)
        
        return answers
    else:
        return [""] * len(prompts)


# =============================================================================
# API compatible interface
# =============================================================================

def local_chat_completion(messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
    """Provides OpenAI-compatible chat completion interface for local model.
    
    This function mimics the OpenAI API structure to provide compatibility
    with existing codebases that expect OpenAI-style responses.
    
    Args:
        messages (List[Dict[str, str]]): List of message dicts with 'role' and 'content'.
        model (str): Model identifier (for compatibility, not used in selection).
        **kwargs: Additional parameters including temperature, max_tokens, response_format.
        
    Returns:
        Dict[str, Any]: OpenAI-compatible response structure with choices, created time, etc.
        
    Note:
        Automatically handles JSON response formatting when response_format type is 'json_object'.
        Uses json_repair for robust JSON parsing and error recovery.
    """
    # Build prompt
    prompt = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "system":
            prompt += content + "\n\n"
        elif role == "user":
            prompt += content
        elif role == "assistant":
            prompt += content + "\n\n"
    
    # Parameters
    temperature = kwargs.get("temperature", 0.7)
    max_tokens = kwargs.get("max_tokens", 512)
    response_format = kwargs.get("response_format", {})
    require_json = response_format.get("type") == "json_object"
    
    # Inference
    answer = simple_inference(prompt, max_tokens, temperature)
    
    # JSON format processing
    if int(os.getenv("RANK", "0")) == 0 and require_json and answer:
        try:
            # Try to repair incomplete JSON
            repaired = repair_json(answer, return_objects=False)
            json.loads(repaired)
            answer = repaired
        except:
            # If repair fails, as a last resort, wrap it
            if not (answer.startswith('{') and answer.endswith('}')):
                escaped_answer = json.dumps(answer)
                answer = f'{{"answer": {escaped_answer}}}'
    
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": answer
            },
            "finish_reason": "stop",
            "index": 0
        }],
        "created": int(time.time()),
        "id": f"local-{int(time.time())}",
        "model": model,
        "object": "chat.completion"
    }


class LocalModelClient:
    """Mock OpenAI client for compatibility with existing API-based code.
    
    This class provides the same interface as OpenAI's AsyncOpenAI client,
    allowing seamless integration with code originally written for API models.
    """
    
    class ChatCompletions:
        """Chat completions interface compatible with OpenAI API."""
        
        def create(self, model: str, messages: List[Dict[str, str]], **kwargs):
            """Creates a chat completion using the local model.
            
            Args:
                model (str): Model identifier.
                messages (List[Dict[str, str]]): Conversation messages.
                **kwargs: Additional completion parameters.
                
            Returns:
                Response object with OpenAI-compatible structure.
            """
            result = local_chat_completion(messages, model, **kwargs)
            
            class Choice:
                def __init__(self, choice_data):
                    self.message = Message(choice_data.get("message", {}))
                    self.finish_reason = choice_data.get("finish_reason", "stop")
                    self.index = choice_data.get("index", 0)
            
            class Message:
                def __init__(self, message_data):
                    self.content = message_data.get("content", "")
                    self.role = message_data.get("role", "assistant")
            
            class Response:
                def __init__(self, response_data):
                    self.choices = [Choice(choice) for choice in response_data.get("choices", [])]
                    self.created = response_data.get("created", int(time.time()))
                    self.id = response_data.get("id", f"local-{int(time.time())}")
                    self.model = response_data.get("model", model)
                    self.object = "chat.completion"
            
            return Response(result)
    
    def __init__(self):
        self.chat = self.ChatCompletions()

def get_local_client():
    """Gets a local model client instance.
    
    Returns:
        LocalModelClient: Client instance for local model operations.
    """
    return LocalModelClient()

# =============================================================================
# Business logic functions
# =============================================================================

def call_llm_local_batch(questions: List[Dict[str, Any]], model_name: str, question_type: str, temperature: float, top_p: float) -> List[Dict[str, Any]]:
    """Performs batch inference on a list of questions using local model.
    
    This function processes multiple questions simultaneously for improved
    GPU utilization and throughput. It handles prompt construction, batch
    inference, and response parsing with comprehensive error handling.
    
    Args:
        questions (List[Dict[str, Any]]): List of question dictionaries containing
            question text, options, context, etc.
        model_name (str): Name of the model (for logging/identification).
        question_type (str): Type of questions (single, multiple, tf, etc.).
        temperature (float): Sampling temperature for generation.
        top_p (float): Top-p (nucleus) sampling parameter.
        
    Returns:
        List[Dict[str, Any]]: List of inference results, each containing:
            - raw_response: Raw model output
            - extracted_answer: Parsed answer from response
            - error: Error message if any occurred
            
    Raises:
        RuntimeError: If model is not initialized.
        
    Note:
        Includes automatic retry logic for failed inferences.
        Handles empty responses and JSON parsing errors gracefully.
    """
    global _engine, _tokenizer
    
    if _engine is None or _tokenizer is None:
        raise RuntimeError("Model not initialized, please call initialize_local_model() first")
    
    if not questions:
        return []
    
    # Build all prompts
    prompts = []
    for q in questions:
        # Get question text
        question_text = q.get("question", "")
        if not question_text:
            prompts.append("")  # Empty prompt, will be handled as error later
            continue
            
        # Build prompt
        user_prompt_parts: List[str] = [f"Question: {str(question_text).strip()}"]
        
        if question_type == "single":
            raw_q_original = str(question_text).strip()
            context_str_bg = q.get("background_text", "")
            if isinstance(context_str_bg, str) and context_str_bg.strip():
                user_prompt_parts.append(f"\nContext:\n{context_str_bg.strip()}")

            options_list_input = q.get("options")
            opts_list_processed: List[str] = []
            if isinstance(options_list_input, list) and options_list_input:
                opts_list_processed = [str(opt).strip() for opt in options_list_input if str(opt).strip()]
            else:
                processed_q_stem, parsed_opts = split_options(raw_q_original)
                if parsed_opts:
                    opts_list_processed = parsed_opts

            if opts_list_processed:
                user_prompt_parts.append("Options:\n" + "\n".join(opts_list_processed))

        elif question_type == "multiple":
            raw_options = q.get("options", [])
            if isinstance(raw_options, list) and all(isinstance(opt, str) for opt in raw_options):
                if raw_options:
                    user_prompt_parts.append("Options:\n" + "\n".join(raw_options))

        # Optional fields
        for field_name, field_label in [
            ("table_html", "Table (HTML)"),
            ("formula_context", "Formula Context"),
            ("question_context", "Additional Context")
        ]:
            field_value = q.get(field_name)
            if field_value and str(field_value).strip():
                if field_name == "table_html":
                    user_prompt_parts.append(f"\n{field_label}:\n```html\n{str(field_value).strip()}\n```")
                else:
                    user_prompt_parts.append(f"\n{field_label}:\n{str(field_value).strip()}")

        user_prompt = "\n\n".join(user_prompt_parts)

        # Select prompt template
        prompt_map = {
            "single": single_prompt,
            "multiple": multiple_prompt,
            "proof": proof_prompt,
            "table": table_prompt,
            "general": general_prompt,
            "numerical": numerical_prompt,
            "fill": fill_prompt,
            "tf": tf_prompt,
        }
        
        if question_type not in prompt_map:
            raise ValueError(f"Unsupported question_type: {question_type}")
        
        system_prompt = prompt_map[question_type]
        
        # Combine system prompt and user prompt
        prompt = system_prompt + "\n\n" + user_prompt
        prompts.append(prompt)
    
    # Batch inference - real batch processing
    results = []
    max_retries = 3
    
    for retry_count in range(max_retries):
        try:
            # Batch encode all prompts
            inputs = _tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
            
            # Move inputs to model device
            if "LOCAL_RANK" in os.environ:
                inputs = inputs.to(_engine.module.device)
            else:
                inputs = inputs.to(_engine.device)
            
            # Batch generate answers
            with torch.no_grad():
                outputs = _engine.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=_tokenizer.eos_token_id
                )
            
            # Batch decode outputs
            responses = _tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Process each response
            results = []
            retry_needed = False
            
            for i, (response, prompt, q) in enumerate(zip(responses, prompts, questions)):
                # Handle empty question case
                if not prompt:
                    results.append({
                        "raw_response": "[\"\", {}]",
                        "extracted_answer": "Error_Empty_Question",
                        "error": "Error: Empty question"
                    })
                    continue
                
                # Extract actual answer (remove prompt part)
                if response.startswith(prompt):
                    actual_response = response[len(prompt):].strip()
                else:
                    actual_response = response.strip()
                
                # Check for empty response
                if actual_response == "[\"\", {}]" or not actual_response:
                    if retry_count < max_retries - 1:
                        retry_needed = True
                        break
                    else:
                        results.append({
                            "raw_response": "[\"\", {}]",
                            "extracted_answer": "Error_Empty_Response",
                            "error": "Error: Empty response after retries"
                        })
                        continue
                
                # Extract answer
                try:
                    # Try to parse JSON
                    if actual_response.startswith('{') and actual_response.endswith('}'):
                        try:
                            answer_json = json.loads(actual_response)
                            if "answer" in answer_json:
                                extracted_answer = answer_json["answer"]
                            else:
                                extracted_answer = "Error_No_Answer_Field"
                        except json.JSONDecodeError:
                            extracted_answer = "Error_Invalid_JSON"
                    else:
                        extracted_answer = "Error_Not_JSON_Format"
                except Exception as e:
                    extracted_answer = f"Error_Extraction_Failed: {str(e)}"
                
                results.append({
                    "raw_response": actual_response,
                    "extracted_answer": extracted_answer,
                    "error": None
                })
            
            # If no retry needed, break out of loop
            if not retry_needed:
                break
                
        except Exception as e:
            if retry_count < max_retries - 1:
                time.sleep(1)  # Wait 1 second before retry
                continue
            else:
                # Last retry failed, return error results
                results = []
                for q in questions:
                    results.append({
                        "raw_response": "[\"\", {}]",
                        "extracted_answer": "Error_LLM_Logic_Failed",
                        "error": f"Error: {str(e)}"
                    })
                break
    
    return results

def call_llm_local(qobj: Dict[str, Any], llm_model: str, question_type: str, temperature: float, top_p: float) -> Dict[str, Any]:
    """Calls local LLM for single question inference.
    
    This function processes a single question using the local model via the
    chat completion interface. It constructs appropriate prompts based on
    question type and handles JSON response parsing.
    
    Args:
        qobj (Dict[str, Any]): Question object containing question text,
            options, context, and other metadata.
        llm_model (str): Model identifier for the local model.
        question_type (str): Type of question (single, multiple, tf, etc.).
        temperature (float): Sampling temperature for generation.
        top_p (float): Top-p (nucleus) sampling parameter.
        
    Returns:
        Dict[str, Any]: Inference result containing:
            - raw_response: Raw model output
            - extracted_answer: Parsed answer from response  
            - error: Error message if any occurred
            
    Note:
        Uses local_chat_completion for OpenAI-compatible interface.
        Includes automatic JSON parsing and validation.
    """
    
    # Build prompt (copy model.py logic)
    user_prompt_parts: List[str] = [f"Question: {str(qobj.get('question')).strip()}"]

    if question_type == "single":
        raw_q_original = str(qobj.get("question")).strip()
        context_str_bg = qobj.get("background_text", "")
        if isinstance(context_str_bg, str) and context_str_bg.strip():
            user_prompt_parts.append(f"\nContext:\n{context_str_bg.strip()}")

        options_list_input = qobj.get("options")
        opts_list_processed: List[str] = []
        if isinstance(options_list_input, list) and options_list_input:
            opts_list_processed = [str(opt).strip() for opt in options_list_input if str(opt).strip()]
        else:
            processed_q_stem, parsed_opts = split_options(raw_q_original)
            if parsed_opts:
                opts_list_processed = parsed_opts

        if opts_list_processed:
            user_prompt_parts.append("Options:\n" + "\n".join(opts_list_processed))

    elif question_type == "multiple":
        raw_options = qobj.get("options", [])
        if isinstance(raw_options, list) and all(isinstance(opt, str) for opt in raw_options):
            if raw_options:
                user_prompt_parts.append("Options:\n" + "\n".join(raw_options))

    # Optional fields
    for field_name, field_label in [
        ("table_html", "Table (HTML)"),
        ("formula_context", "Formula Context"),
        ("question_context", "Additional Context")
    ]:
        field_value = qobj.get(field_name)
        if field_value and str(field_value).strip():
            if field_name == "table_html":
                user_prompt_parts.append(f"\n{field_label}:\n```html\n{str(field_value).strip()}\n```")
            else:
                user_prompt_parts.append(f"\n{field_label}:\n{str(field_value).strip()}")

    user_prompt = "\n\n".join(user_prompt_parts)

    # Select prompt template
    prompt_map = {
        "single": single_prompt,
        "multiple": multiple_prompt,
        "proof": proof_prompt,
        "table": table_prompt,
        "general": general_prompt,
        "numerical": numerical_prompt,
        "fill": fill_prompt,
        "tf": tf_prompt,
    }
    
    if question_type not in prompt_map:
        raise ValueError(f"Unsupported question_type: {question_type}")
    
    prompt = prompt_map[question_type]
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Token limits
    if question_type in ["single", "multiple", "numerical"]:
        max_tokens = 60
    elif question_type in ["tf"]:
        max_tokens = 150
    else:
        max_tokens = 512

    # Directly call local chat completion function
    try:
        response_data = local_chat_completion(
            messages=messages,
            model=llm_model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        
        # Extract response content
        raw_content = ""
        if response_data.get("choices") and len(response_data["choices"]) > 0:
            raw_content = response_data["choices"][0].get("message", {}).get("content", "")

        if raw_content and UPSTREAM_SATURATION_KEYWORD in raw_content:
            raise LocalModelError(f"Local model overload: {raw_content[:200]}…")

        # JSON parsing
        extracted_answer = "Error_LLM_Logic_Failed"
        err = None
        
        if int(os.getenv("RANK", "0")) == 0:
            if not raw_content:
                err = "Error: Local model returned empty content"
            else:
                try:
                    parsed_json = json.loads(raw_content)
                except:
                    try:
                        repaired_json_string = repair_json(raw_content)
                        parsed_json = json.loads(repaired_json_string)
                        err = "Warning: Original JSON was invalid and repaired with json_repair."
                    except Exception as repair_exc:
                        err = f"Error: Failed after json_repair -> {repair_exc}. Raw response: '{raw_content[:100]}…'"
                        parsed_json = None

                # Validate answer field
                if parsed_json and isinstance(parsed_json, dict):
                    if "answer" in parsed_json:
                        extracted_answer = parsed_json["answer"]
                        err = None

                        # Type validation
                        if question_type == "single":
                            ans_val = str(extracted_answer).strip().upper()
                            if ans_val in {"A", "B", "C", "D"}:
                                extracted_answer = ans_val
                            else:
                                err = f"Error: 'answer' value '{extracted_answer}' is not A, B, C, or D."
                                extracted_answer = "Error_Invalid_Option_Value"

                        elif question_type == "multiple":
                            if isinstance(extracted_answer, list):
                                processed_selection = [str(item).strip().upper() for item in extracted_answer if str(item).strip()]
                                extracted_answer = sorted(list(set(processed_selection)))
                            else:
                                err = "Error: 'answer' key must be a list for 'multiple' question_type."
                                extracted_answer = "Error_Invalid_List"

                        elif question_type == "tf":
                            if isinstance(extracted_answer, str):
                                ans_upper = extracted_answer.upper()
                                if not (ans_upper.startswith("TRUE.") or ans_upper.startswith("FALSE.") or
                                      "INSUFFICIENT" in ans_upper or "CANNOT DETERMINE" in ans_upper or
                                      "NOT ENOUGH INFORMATION" in ans_upper):
                                    err = "Error: 'answer' string format is not as expected for TF question_type."
                            else:
                                err = "Error: 'answer' key must be a string for TF question_type."
                    else:
                        err = "Error: JSON is missing the 'answer' key."
                elif not err:
                    err = "Error: The response from local model is not a valid JSON object."

    except Exception as e:
        err = f"Error: Unexpected exception during local model interaction. {e}"
        print(f"Unexpected exception during local model interaction Q: '{str(qobj.get('question'))[:50]}…' (Model: {llm_model}, Temp: {temperature}, Top_p:{top_p}): {e}.")
        raw_content = ""
        extracted_answer = "Error_Unexpected_Exception"

    return {
        "raw_response": raw_content or "",
        "extracted_answer": extracted_answer,
        "error": err,
    }

def process_question_item_local(qobj: Dict[str, Any], llm_model: str, question_type: str, temperature: float, top_p: float) -> Optional[Dict[str, Any]]:
    """Processes a single question item through local model inference.
    
    This function wraps call_llm_local to provide a complete question
    processing workflow, including result formatting compatible with
    the evaluation framework.
    
    Args:
        qobj (Dict[str, Any]): Question object to process.
        llm_model (str): Model identifier.
        question_type (str): Type of question.
        temperature (float): Sampling temperature.
        top_p (float): Top-p parameter.
        
    Returns:
        Optional[Dict[str, Any]]: Processed record with model evaluation results,
            or None if processing failed.
    """
    llm_result = call_llm_local(qobj, llm_model, question_type, temperature, top_p)
    processed_record = dict(qobj)
    processed_record["model_evaluation_result"] = {
        "model_raw_response": llm_result["raw_response"],
        "model_answer": llm_result["extracted_answer"],
        "error": llm_result["error"],
    }
    return processed_record

def run_batch_local(task_configs: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Runs complete batch evaluation tasks across all distributed processes.
    
    This function coordinates the entire distributed evaluation workflow:
    1. Broadcasts task configurations from rank 0 to all processes
    2. Processes each task sequentially across all domains/question types
    3. Handles data loading, model inference, and result saving
    4. Provides comprehensive error handling and progress reporting
    
    Args:
        task_configs (List[Dict[str, Any]]): List of task configuration dictionaries,
            each containing domain, question_type, model_name, temperature, top_p,
            dataset_path, and output_path.
            
    Returns:
        Tuple[int, int]: (successful_tasks, failed_tasks) count.
        
    Note:
        This function is called by all processes in distributed mode.
        Only rank 0 performs file I/O operations and progress reporting.
        Uses distributed barriers for synchronization between processes.
    """
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    is_main_process = (rank == 0)

    # --- 1. Rank 0 broadcast task configuration ---
    if world_size > 1:
        bcast_obj_list = [task_configs] if is_main_process else [None]
        torch.distributed.broadcast_object_list(bcast_obj_list, src=0)
        if not is_main_process:
            task_configs = bcast_obj_list[0]

    if not task_configs and is_main_process:
        print("Warning: No task configuration received.")
        return 0, 0

    successful_tasks = 0
    failed_tasks = 0
    start_time = time.time()

    # --- 2. All processes process each task sequentially ---
    for i, config in enumerate(task_configs):
        if is_main_process:
            print(f"\n{'='*60}")
            print(f"📈 Progress: {i+1}/{len(task_configs)}")
            print(f"🚀 Running: LOCAL | {config['domain']} | {config['question_type']} | T:{config['temperature']} | Top_p:{config['top_p']}")
            print(f"{'='*60}")
        
        task_start_time = time.time()
        
        try:
            # --- a. Prepare paths and data ---
            # Map question_type to actual dataset directory name
            from main import QUESTION_TYPE_MAP
            actual_question_type = QUESTION_TYPE_MAP.get(config['question_type'], config['question_type'])
            input_dir = Path(config['dataset_path']) / config['domain'] / actual_question_type
            output_dir = Path(config['output_path']) / config['domain'] / config['question_type'] / config['model_name'] / f"tem{config['temperature']}" / f"top_k{config['top_p']}"
            eval_dir = output_dir / "evaluation"
            
            if is_main_process:
                print(f"  📂 Model name: {config['model_name']}")
                print(f"  📂 Output directory: {output_dir}")
                print(f"  📂 Evaluation directory: {eval_dir}")

            json_files_to_process = []
            if is_main_process:
                output_dir.mkdir(parents=True, exist_ok=True)
                eval_dir.mkdir(parents=True, exist_ok=True)
                json_files_to_process = sorted([str(f) for f in input_dir.glob("*.json") if f.is_file()])
            
            if world_size > 1:
                torch.distributed.barrier()
                bcast_files = [json_files_to_process]
                torch.distributed.broadcast_object_list(bcast_files, src=0)
                if not is_main_process:
                    json_files_to_process = bcast_files[0]
            
            # --- b. Loop through files ---
            for file_path_str in json_files_to_process:
                input_json_file_path = Path(file_path_str)
                if is_main_process:
                    print(f"  Processing file: {input_json_file_path.name}")
                
                eval_jsonl_path = eval_dir / f"{input_json_file_path.stem}_eval.jsonl"
                
                # Rank 0 loads and broadcasts questions
                questions_to_process_now = []
                all_records_for_this_file = []
                if is_main_process:
                    # Read processed questions
                    processed_records = []
                    if eval_jsonl_path.exists():
                        with eval_jsonl_path.open("r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    record = json.loads(line)
                                    # Check if it's an empty response
                                    if record.get("model_evaluation_result", {}).get("model_raw_response") == "[\"\", {}]":
                                        continue  # Skip empty responses, need to reprocess
                                    processed_records.append(record)
                                except json.JSONDecodeError:
                                    continue
                    
                    # Read all questions
                    with input_json_file_path.open("r", encoding="utf-8") as f:
                        all_questions = json.load(f)
                    
                    # Get processed question IDs
                    processed_qids = {r.get("qid") for r in processed_records if r.get("qid")}
                    
                    # Filter out unprocessed questions
                    questions_to_process_now = [
                        q for q in all_questions 
                        if q.get("qid") not in processed_qids
                    ]
                    
                    all_records_for_this_file.extend(processed_records)

                if world_size > 1:
                    bcast_questions = [questions_to_process_now]
                    torch.distributed.broadcast_object_list(bcast_questions, src=0)
                    if not is_main_process:
                        questions_to_process_now = bcast_questions[0]
                    torch.distributed.barrier()
                
                if not questions_to_process_now:
                    if is_main_process: print("    [INFO] No new questions to process.")
                    continue
                
                pbar = None
                if is_main_process:
                    pbar = tqdm(total=len(questions_to_process_now), desc=f"  Inference {input_json_file_path.stem}", unit="q", ncols=120)
                
                # Batch inference - process multiple questions at once to improve GPU utilization
                batch_size = BATCH_SIZE
                for i in range(0, len(questions_to_process_now), batch_size):
                    batch = questions_to_process_now[i:i + batch_size]
                    
                    # Batch call local model
                    batch_results = call_llm_local_batch(
                        batch, config['model_name'], config['question_type'], config['temperature'], config['top_p']
                    )
                    
                    # Process batch results
                    for q_obj, llm_result in zip(batch, batch_results):
                        processed_record = dict(q_obj)
                        processed_record["model_evaluation_result"] = {
                            "model_raw_response": llm_result["raw_response"],
                            "model_answer": llm_result["extracted_answer"],
                            "error": llm_result["error"],
                        }
                        
                        if is_main_process:
                            if processed_record:
                                all_records_for_this_file.append(processed_record)
                                with eval_jsonl_path.open("a", encoding="utf-8") as f_jsonl:
                                    f_jsonl.write(json.dumps(processed_record, ensure_ascii=False) + "\n")
                            if pbar: pbar.update(1)
                
                if pbar: pbar.close()
                
                # --- c. Rank 0 saves final results ---
                if is_main_process:
                    final_unique_records_map = {rec.get('qid'): rec for rec in all_records_for_this_file if rec.get('qid')}
                    eval_json_path = eval_dir / f"{input_json_file_path.stem}_eval.json"
                    eval_json_path.write_text(json.dumps(list(final_unique_records_map.values()), ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"    ✅ Results saved to: {eval_json_path.name}")
            
            if world_size > 1:
                torch.distributed.barrier()

            if is_main_process:
                duration = time.time() - task_start_time
                print(f"✅ Task completed ({duration:.1f}s)")
            successful_tasks += 1

        except Exception as e:
            if is_main_process:
                duration = time.time() - task_start_time
                print(f"❌ Task failed ({duration:.1f}s): {str(e)}")
                import traceback
                traceback.print_exc()
            failed_tasks += 1
    
    if is_main_process:
        total_time = time.time() - start_time
        print(f"\n⏱️  Local batch run total time: {total_time/60:.1f} minutes")

    return successful_tasks, failed_tasks


def process_directory_local(*args, **kwargs):
    """Deprecated function maintained for backward compatibility.
    
    This function was replaced by run_batch_local for improved distributed
    processing capabilities. It raises a deprecation warning when called.
    
    Args:
        *args: Variable arguments (ignored).
        **kwargs: Keyword arguments (ignored).
        
    Raises:
        DeprecationWarning: Always raised to indicate deprecated status.
        
    Note:
        Use run_batch_local instead for new implementations.
    """
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        print("Warning: Called deprecated process_directory_local function.", file=sys.stderr)
    raise DeprecationWarning("process_directory_local is deprecated. Use run_batch_local instead.")


# Async version (for evaluation.py)
async def local_chat_completion_async(messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
    """Provides asynchronous wrapper for local chat completion.
    
    This function enables async compatibility with the evaluation system
    by wrapping the synchronous local_chat_completion in a thread executor.
    
    Args:
        messages (List[Dict[str, str]]): Conversation messages.
        model (str): Model identifier.
        **kwargs: Additional completion parameters.
        
    Returns:
        Dict[str, Any]: Chat completion response in OpenAI format.
        
    Note:
        Used by evaluation.py for async LLM grading with local models.
        Runs synchronous function in thread pool to avoid blocking.
    """
    import asyncio
    import concurrent.futures
    
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        result = await loop.run_in_executor(executor, lambda: local_chat_completion(messages, model, **kwargs))
    return result

def check_local_model_health():
    """Checks if local model is properly initialized and available.
    
    Returns:
        bool: True if model is initialized and ready for inference, False otherwise.
        
    Note:
        Used by evaluation.py to verify model availability before processing.
        Does not perform actual inference, only checks initialization status.
    """
    return _is_initialized 