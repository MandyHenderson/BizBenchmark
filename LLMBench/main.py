"""Main entry point for BizBenchmark evaluation framework.

This module provides command-line interface for batch evaluation of large language models
on business decision-making tasks across four domains: Economics (ECON), Finance (FIN),
Operations Management (OM), and Statistics (STAT).

The script supports two execution modes:
    - API mode: Uses OpenAI-compatible API for model inference
    - Local mode: Uses locally deployed models with DeepSpeed support

Example:
    Basic usage:
        $ python main.py --domain ECON --question_type single --model deepseek-chat

    Batch processing:
        $ python main.py --batch_all --model_type local

    Multi-GPU local inference:
        $ torchrun --nproc_per_node=4 main.py --model_type local

Typical usage example:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run(args)
"""
import argparse
import os
import time
from itertools import product
import sys

parser = argparse.ArgumentParser()

# Batch execution configuration
ALL_DOMAINS = ['ECON', 'FIN', 'OM', 'STAT']
ALL_QUESTION_TYPES = ['fill', 'general', 'multiple', 'numerical', 'proof', 'single', 'table', 'tf']

# Question type mapping: command line argument -> dataset directory name
QUESTION_TYPE_MAP = {
    'single': 'Single_Choice',
    'multiple': 'Multiple_Choice', 
    'tf': 'TF',
    'fill': 'Fill-in-the-Blank',
    'general': 'General_QA',
    'numerical': 'Numerical_QA',
    'proof': 'Proof',
    'table': 'Table_QA'
}


def run_single_evaluation(domain, question_type, model_name, temperature, top_p, model_type, dataset_path, output_path):
    """Executes single evaluation task for specified parameters.

    This function handles both API and local model evaluation modes. For API mode,
    it directly calls the process_directory function. For local mode, it builds
    task configuration and uses batch processing infrastructure.

    Args:
        domain (str): Business domain (ECON, FIN, OM, STAT).
        question_type (str): Question type identifier (single, multiple, tf, etc.).
        model_name (str): Name of the LLM model to evaluate.
        temperature (float): Sampling temperature for model inference.
        top_p (float): Nucleus sampling parameter.
        model_type (str): Model execution type ('api' or 'local').
        dataset_path (str): Path to the dataset directory.
        output_path (str): Directory for saving evaluation results.

    Returns:
        bool: True if evaluation completed successfully, False otherwise.

    Note:
        For local mode, this function is primarily used for logging and maintaining
        compatibility with single-task execution patterns.
    """
    is_main_process = int(os.getenv("RANK", "0")) == 0
    # Map question_type to actual dataset directory name
    actual_question_type = QUESTION_TYPE_MAP.get(question_type, question_type)
    input_path = os.path.join(dataset_path, domain, actual_question_type)
    result_path = os.path.join(output_path, domain, question_type, model_name, f'tem{temperature}', f'top_k{top_p}')
    
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"🚀 Running: {model_type.upper()} | {domain} | {question_type} | T:{temperature} | Top_p:{top_p}")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # API mode remains unchanged
        if model_type == "api":
            from model.model import process_directory
            process_directory(input_path, result_path, question_type, model_name, temperature, top_p)
            duration = time.time() - start_time
            if is_main_process:
                print(f"✅ Task completed ({duration:.1f}s)")
            return True
        elif model_type == "local":
            # Local mode single task processing
            from model.local_model import run_batch_local
            # Build single task configuration
            task_config = {
                "domain": domain,
                "question_type": question_type,
                "model_name": model_name,
                "temperature": temperature,
                "top_p": top_p,
                "dataset_path": dataset_path,
                "output_path": output_path
            }
            total_success, total_fail = run_batch_local([task_config])
            duration = time.time() - start_time
            if is_main_process:
                if total_success > 0:
                    print(f"✅ Task completed ({duration:.1f}s)")
                else:
                    print(f"❌ Task failed ({duration:.1f}s)")
            return total_success > 0
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
    except Exception as e:
        duration = time.time() - start_time
        if is_main_process:
            print(f"❌ Task failed ({duration:.1f}s): {str(e)}")
        return False


def run_batch_all(args):
    """Executes batch evaluation across multiple domains and question types.

    This function generates all possible combinations of domains and question types
    based on provided arguments, then executes them either sequentially (API mode)
    or in parallel (local mode).

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
            - domains: List of domains to evaluate (default: all domains)
            - question_types: List of question types (default: all types)
            - model_type: Execution mode ('api' or 'local')
            - model: Model name
            - temperature: Sampling temperature
            - top_p: Nucleus sampling parameter
            - dry_run: If True, only preview tasks without execution
            - continue_on_error: If True, continue batch execution on errors

    Note:
        For local mode, all tasks are executed simultaneously using distributed
        processing. For API mode, tasks are executed sequentially with optional
        error handling.
    """
    # Generate all task combinations
    tasks = list(product(
        args.domains if args.domains else ALL_DOMAINS,
        args.question_types if args.question_types else ALL_QUESTION_TYPES,
    ))
    
    is_main_process = int(os.getenv("RANK", "0")) == 0
    
    if is_main_process:
        print(f"📋 Batch run mode - Total {len(tasks)} tasks:")
        print(f"   Model type: {args.model_type}")
        print(f"   Domains: {args.domains if args.domains else ALL_DOMAINS}")
        print(f"   Question types: {args.question_types if args.question_types else ALL_QUESTION_TYPES}")
        print(f"   Model: {args.model}")
        print(f"   Temperature: {args.temperature}")
        print(f"   Top_p: {args.top_p}")
    
    if args.dry_run:
        if is_main_process:
            print("\n🔍 Task preview (dry run):")
            for i, (domain, qtype) in enumerate(tasks, 1):
                print(f"  {i:2d}. {args.model_type.upper()} | {domain} | {qtype} | T:{args.temperature} | Top_p:{args.top_p}")
        return
    
    # Local mode: Execute all tasks at once
    if args.model_type == 'local':
        from model.local_model import run_batch_local
        
        # Main process prepares task configuration list
        task_configs = []
        if is_main_process:
            for domain, qtype in tasks:
                task_configs.append({
                    "domain": domain,
                    "question_type": qtype,
                    "model_name": args.model,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "dataset_path": args.dataset_path,
                    "output_path": args.output_path
                })
        
        # All processes call the same function, passing configuration
        # run_batch_local handles distributed logic internally
        total_success, total_fail = run_batch_local(task_configs)

        if is_main_process:
            print(f"\n{'='*60}")
            print(f"📊 Batch run completed")
            print(f"✅ Success: {total_success}")
            print(f"❌ Failed: {total_fail}")
            print(f"{'='*60}")
        return

    # API mode: Keep original logic
    if is_main_process and not args.continue_on_error:
        print(f"\n⚠️  About to run {len(tasks)} tasks")
        print("⚠️  Will stop on error (add --continue_on_error to continue)")
        
        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ Cancelled")
            return
    elif is_main_process:
        print("\n🚀 Detected --continue_on_error, starting automatically...")

    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, (domain, question_type) in enumerate(tasks, 1):
        if is_main_process:
            print(f"\n📈 Progress: {i}/{len(tasks)}")
        
        success = run_single_evaluation(
            domain, question_type, args.model, args.temperature, args.top_p, args.model_type,
            args.dataset_path, args.output_path
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            if not args.continue_on_error:
                if is_main_process:
                    print(f"\n💥 Stopped at task {i} (--continue_on_error not enabled)")
                break
    
    if is_main_process:
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"📊 Batch run completed")
        print(f"✅ Success: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"{'='*60}")


def run_single(args):
    """Executes single evaluation task based on command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
            evaluation parameters.

    Raises:
        SystemExit: If evaluation fails, exits with code 1.
    """
    success = run_single_evaluation(
        args.domain, args.question_type, args.model, args.temperature, args.top_p, args.model_type,
        args.dataset_path, args.output_path
    )
    if not success:
        exit(1)


def run(args):
    """Main execution function that initializes models and coordinates evaluation.

    This function handles the complete evaluation workflow:
    1. Initializes local model if required
    2. Executes batch or single evaluation based on arguments
    3. Cleans up distributed resources

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Note:
        For local model mode, this function performs one-time model initialization
        that must occur at the top level before any processing begins.
    """
    # Critical: One-time initialization before any processing begins at the top level of the main script
    if args.model_type == 'local':
        from model.local_model import initialize_local_model
        rank = int(os.getenv("RANK", "0"))
        if rank == 0:
            print("============================================================")
            print("🚀 Local Model Mode: Initializing model from main.py...")
            print("============================================================")
        try:
            initialize_local_model()
            if rank == 0:
                print("✅ Model initialized successfully by main.py.")
        except Exception as e:
            if rank == 0:
                print(f"❌ FATAL: Local model initialization failed: {e}")
                import traceback
                traceback.print_exc()
            sys.exit(1)

    try:
        if args.batch_all:
            run_batch_all(args)
        else:
            # Single task mode
            is_main_process = int(os.getenv("RANK", "0")) == 0
            if is_main_process:
                print(f"🌐 Using {'Local' if args.model_type == 'local' else 'API'} mode")
            run_single(args)
    finally:
        # Clean up process group before program ends if in distributed mode
        if args.model_type == 'local' and int(os.getenv('WORLD_SIZE', '1')) > 1:
            import torch.distributed
            if torch.distributed.is_initialized():
                rank = int(os.getenv("RANK", "0"))
                if rank == 0:
                    print("\n🧹 Cleaning up distributed process group...")
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser.add_argument('--dataset_path', default='../Dataset', type=str, help='Directory containing all domains')
    parser.add_argument('--domain', default='ECON', type=str, help='Business domain: (ECON, FIN, OM, STAT)')
    parser.add_argument('--output_path', default='./result', type=str, help='Output directory')
    parser.add_argument('--model', default='deepseek-chat', type=str, help='Name of LLM model')
    parser.add_argument('--question_type', default='tf', type=str, help='Type of choice: (fill, general, multiple, numerical, proof, single, table, tf)')
    parser.add_argument('--temperature', default=0.2, type=float, help='temperature of the LLM')
    parser.add_argument('--top_p', default=0.95, type=float, help='top of the LLM')
    parser.add_argument('--model_type', default='api', type=str, choices=['api', 'local'], help='Model type: api for OpenAI-like API, local for local deployed model')
    
    # Batch run options
    parser.add_argument('--batch_all', action='store_true', help='Batch run all domain and question type combinations')
    parser.add_argument('--domains', nargs='+', choices=ALL_DOMAINS, help='Specify domains for batch run (default: all)')
    parser.add_argument('--question_types', nargs='+', choices=ALL_QUESTION_TYPES, help='Specify question types for batch run (default: all)')
    parser.add_argument('--continue_on_error', action='store_true', help='Continue on error during batch run')
    parser.add_argument('--dry_run', action='store_true', help='Only preview task list, do not actually run')
    
    args = parser.parse_args()
    run(args)