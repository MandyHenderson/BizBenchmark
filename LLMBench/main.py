"""main.py

Command‑line entry point for batch evaluation.

The script parses arguments such as dataset path, domain, model name, question type,
and sampling parameters, then calls ``process_directory`` from
``model.model`` to run the evaluation.
"""
import argparse
import os
from model.model import process_directory
parser = argparse.ArgumentParser()
def run(args):
    dataset_path = args.dataset_path
    domain = args.domain
    model_name = args.model
    question_type = args.question_type
    input_path = os.path.join(dataset_path, domain, question_type)
    temperature = args.temperature
    top_p = args.top_p
    output_path = os.path.join(args.output_path, domain, question_type, model_name, f'tem{temperature}', f'top_k{top_p}')
    process_directory(input_path, output_path, question_type, model_name, temperature, top_p)

if __name__ == "__main__":
    parser.add_argument('--dataset_path', default='./dataset', type=str, help='Directory containing all domains')
    parser.add_argument('--domain', default='ECON', type=str, help='Business domain: (ECON, FIN, OM, STAT)')
    parser.add_argument('--output_path', default='./result', type=str, help='Output directory')
    parser.add_argument('--model', default='deepseek-chat', type=str, help='Name of LLM model')
    parser.add_argument('--question_type', default='tf', type=str, help='Type of choice: (fill, general, multiple, numerical, proof, single, table, tf)')
    parser.add_argument('--temperature', default=0.2, type=float, help='temperature of the LLM')
    parser.add_argument('--top_p', default=0.95, type=float, help='top of the LLM')
    args = parser.parse_args()
    run(args)