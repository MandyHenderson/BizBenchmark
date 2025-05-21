import argparse
import os
from model.model import process_directory
parser = argparse.ArgumentParser()
def run(args):
    dataset_path = args.dataset_path
    model_name = args.model
    question_type = args.question_type
    input_path = os.path.join(dataset_path, question_type)
    temperature = args.temperature
    top_p = args.top_p
    output_path = os.path.join(args.output_path, question_type, model_name, f'tem{temperature}', f'top_k{top_p}')
    process_directory(input_path, output_path, question_type, model_name, temperature, top_p)

if __name__ == "__main__":
    parser.add_argument('--dataset_path', default='./dataset', type=str, help='Directory containing all question')
    parser.add_argument('--output_path', default='./result', type=str, help='Output directory')
    parser.add_argument('--model', default='deepseek-chat', type=str, help='Name of LLM model')
    parser.add_argument('--question_type', default='tf', type=str, help='Type of chioce: (fill, general, multiple, numerical, proof, single, table, tf)')
    parser.add_argument('--temperature', default=0.2, type=float, help='temperature of the LLM')
    parser.add_argument('--top_p', default=0.95, type=float, help='top of the LLM')
    args = parser.parse_args()
    run(args)