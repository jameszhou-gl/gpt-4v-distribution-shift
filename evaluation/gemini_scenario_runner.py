"""
This script evaluates Gemini across different scenarios (failure and random) and combines the results from each scenario for analysis. 
It assumes the presence of four directories: failure_1, failure_2, random_1, random_2, random_3, random_4, each containing the relevant data for evaluation.
"""

import os
import argparse
import json
import subprocess
from evaluation.utils import analyze_combined_result


def load_data_info():
    """ Load dataset information from the dataset_info.json file. """
    with open('./data/dataset_info.json', 'r') as file:
        return json.load(file)


def load_jsonl(file_path):
    """ Load data from a JSONL file. """
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def check_directories_in_list(base_directory):
    """
    Check if only one dataset is present in the specified directory.
    Raise an error if more than one dataset is found.
    """
    subdirectories = [name for name in os.listdir(base_directory)
                      if os.path.isdir(os.path.join(base_directory, name))]
    dataset_list = list(data_info.keys())
    matches = [name for name in subdirectories if name in dataset_list]
    if len(matches) > 1:
        raise ValueError(
            "Only one dataset is allowed for evaluation with Gemini at the same time.")
    return matches[0]


def combine_jsonl_files(file_paths):
    """
    Combine multiple JSONL files into a single list.
    """
    combined_data = []
    for file_path in file_paths:
        data = load_jsonl(file_path)
        combined_data.extend(data)
    return combined_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Gemini on two scenarios with four directories: failure_1, failure_2, random_1, random_2.')
    parser.add_argument('--model_name', type=str,
                        choices=['gemini-pro-vision'], default="gemini-pro-vision")
    parser.add_argument('--continue_dir', type=str, required=True,
                        help='Directory containing the CLIP sample sets, e.g., "exp_output/2023-11-18-19_56_06".')
    parser.add_argument('--scenario_name', type=str, required=True,
                        choices=['failure_1', 'failure_2', 'random_1', 'random_2', 'random_3', 'random_4'])
    parser.add_argument('--GOOGLE_API_KEY', type=str, required=True)

    args = parser.parse_args()
    data_info = load_data_info()
    dataset = check_directories_in_list(args.continue_dir)
    fold_dir = os.path.join(args.continue_dir, args.scenario_name)

    # Run Gemini evaluation for the specified scenario
    eval_gemini_single = [
        "python", "-m", "evaluation.eval_gemini",
        "--continue_dir", f"{fold_dir}/{dataset}",
        "--dataset", dataset,
        "--GOOGLE_API_KEY", args.GOOGLE_API_KEY
    ]
    subprocess.run(eval_gemini_single)

    # Combine and analyze results for the final scenarios
    if args.scenario_name == 'random_2':
        scenario_type = 'random'
        file_paths = [
            f'{args.continue_dir}/{scenario_type}_1/{dataset}/unified_output_{args.model_name}.jsonl',
            f'{args.continue_dir}/{scenario_type}_2/{dataset}/unified_output_{args.model_name}.jsonl'
        ]
        combined_data = combine_jsonl_files(file_paths)
        analyze_combined_result(
            combined_data, scenario_type, args.continue_dir, args.model_name)
    elif args.scenario_name == 'random_4':
        scenario_type = 'random'
        file_paths = [
            f'{args.continue_dir}/{scenario_type}_1/{dataset}/unified_output_{args.model_name}.jsonl',
            f'{args.continue_dir}/{scenario_type}_2/{dataset}/unified_output_{args.model_name}.jsonl',
            f'{args.continue_dir}/{scenario_type}_3/{dataset}/unified_output_{args.model_name}.jsonl',
            f'{args.continue_dir}/{scenario_type}_4/{dataset}/unified_output_{args.model_name}.jsonl'
        ]
        combined_data = combine_jsonl_files(file_paths)
        analyze_combined_result(
            combined_data, scenario_type, args.continue_dir, args.model_name)
    elif args.scenario_name == 'failure_2':
        scenario_type = 'failure'
        file_paths = [
            f'{args.continue_dir}/{scenario_type}_1/{dataset}/unified_output_{args.model_name}.jsonl',
            f'{args.continue_dir}/{scenario_type}_2/{dataset}/unified_output_{args.model_name}.jsonl'
        ]
        combined_data = combine_jsonl_files(file_paths)
        analyze_combined_result(
            combined_data, scenario_type, args.continue_dir, args.model_name)
    else:
        pass
