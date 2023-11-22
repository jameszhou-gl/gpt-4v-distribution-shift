"""
This script prepares data for GPT-4V evaluation based on the results from CLIP and LLaVA models. It processes both failure and random cases, and arranges the data for further evaluation.

The script performs the following steps:
1. Extracts a specified number of failure and random cases from the CLIP results.
2. Processes the results for both CLIP and LLaVA models.
3. Splits the cases into different scenarios (failure and random) and organizes them into appropriate directories.
"""

import os
import argparse
import random
import json
import shutil
from evaluation.utils import analyze_combined_result


def load_jsonl(file_path):
    """ Load data from a JSONL file. """
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def load_data_info():
    """ Load dataset information from a JSON file. """
    with open('./data/dataset_info.json', 'r') as file:
        return json.load(file)


def load_clip_results(args):
    """ Load CLIP evaluation results from a file. """
    unified_output_clip = os.path.join(
        args.continue_dir, f'unified_output_{args.clip_model}.jsonl')
    return load_jsonl(unified_output_clip)


def extract_failure_cases(clip_results, k):
    """ Extract failure cases where the predicted class does not match the true class. """
    failures = [case for case in clip_results if case['true_class']
                != case['predicted_class']]
    num_failures = len(failures)
    sample_size = min(k, num_failures)
    print(f'Total failure cases: {num_failures}; We sample: {sample_size}')
    return random.sample(failures, sample_size)


def extract_random_samples(clip_results, k):
    """ Extract a random subset of samples from CLIP results. """
    total_cases = len(clip_results)
    sample_size = min(k, total_cases)
    print(f'Total cases: {total_cases}; We sample: {sample_size}')
    return random.sample(clip_results, sample_size)


def split_into_folds(cases, num_folds=2):
    """ Split cases into the specified number of folds. """
    fold_size = len(cases) // num_folds
    return [cases[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]


def create_directories_and_copy_files(cases, base_dir, fold_name):
    """ Create directories for each fold and copy files into them. """
    print(f"Processing {len(cases)} cases in {fold_name}")
    domains = []
    class_names = []
    selected_images_info = {'dataset': '', 'domains': [],
                            'class_names': [], 'samples': {}}
    for idx, case in enumerate(cases):
        # Extract domain, class, and image file path
        domain = case['domain']
        class_name = case['true_class']
        dataset = case['dataset']
        img_path = os.path.join(base_dir, dataset, case['image'])
        filename = os.path.basename(img_path)

        # Create directory structure
        dir_path = os.path.join(base_dir, fold_name,
                                dataset, domain, class_name)
        os.makedirs(dir_path, exist_ok=True)

        # Copy file to the new directory
        shutil.copy(img_path, os.path.join(dir_path, filename))
        domains.append(domain)
        class_names.append(class_name)
        selected_images_info['samples'][str(idx)] = {
            "domain": domain,
            "class": class_name,
            "image": case['image'],
            "subject": data_info[dataset]['subject']
        }
    selected_images_info['dataset'] = dataset
    selected_images_info['domains'] = list(set(domains))
    selected_images_info['class_names'] = list(set(class_names))
    with open('{}/unified_input_{}.json'.format(os.path.join(base_dir, fold_name, dataset), dataset), 'w') as f:
        json.dump(selected_images_info, f, indent=4)
    # with open('{}/log.txt'.format(os.path.join(base_dir, fold_name, dataset)), 'w') as f:
    #     f.write('')


def check_directories_in_list(base_directory):
    """ Check if more than one dataset is present in the specified directory. """
    subdirectories = [name for name in os.listdir(base_directory)
                      if os.path.isdir(os.path.join(base_directory, name))]
    dataset_list = list(data_info.keys())
    # Check if more than one sub-directory name is in the dataset_list
    matches = [name for name in subdirectories if name in dataset_list]

    # Raise an error if more than one match is found
    if len(matches) > 1:
        raise ValueError(
            "Only one dataset is allowed for evaluation with GPT-4V at the same time.")
    return matches[0]


def prepare_results_for_clip_llava(failure_cases, random_samples, args):
    """ Prepare and analyze results for CLIP and LLaVA models. """
    clip_output_path = f'{args.continue_dir}/unified_output_{args.clip_model}.jsonl'
    llava_output_path = f'{args.continue_dir}/unified_output_{args.llava_model}.jsonl'

    # Load data from files
    try:
        clip_output = load_jsonl(clip_output_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"CLIP output file not found: {clip_output_path}")

    try:
        llava_output = load_jsonl(llava_output_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"LLaVA output file not found: {llava_output_path}")

    # Process failure cases
    clip_fail_cases = [clip_output[int(failure_case['id']) - 1]
                       for failure_case in failure_cases]
    llava_fail_cases = [
        llava_output[int(failure_case['id']) - 1] for failure_case in failure_cases]
    analyze_combined_result(clip_fail_cases, 'failure',
                            args.continue_dir, args.clip_model)
    analyze_combined_result(llava_fail_cases, 'failure',
                            args.continue_dir, args.llava_model)

    # Process random samples
    clip_rand_cases = [clip_output[int(random_sample['id']) - 1]
                       for random_sample in random_samples]
    llava_rand_cases = [
        llava_output[int(random_sample['id']) - 1] for random_sample in random_samples]
    analyze_combined_result(clip_rand_cases,
                            'random', args.continue_dir, args.clip_model)
    analyze_combined_result(llava_rand_cases, 'random',
                            args.continue_dir, args.llava_model)


def prepare_data_for_gpt4v(args):
    """ Main function to prepare data for GPT-4V evaluation. """
    clip_results = load_clip_results(args)
    # Extract and split failure cases
    failure_cases = extract_failure_cases(clip_results, args.num_failure)
    # Extract and split random samples
    random_samples = extract_random_samples(clip_results, args.num_rand)
    # we manage the results for clip and llava in each scenario: failure and rand
    prepare_results_for_clip_llava(failure_cases, random_samples, args)
    # Split into folds
    failure_folds = split_into_folds(failure_cases)
    random_folds = split_into_folds(random_samples)
    # Process each fold
    for i, fold in enumerate(failure_folds, start=1):
        create_directories_and_copy_files(
            fold, args.continue_dir, f'failure_{i}')

    for i, fold in enumerate(random_folds, start=1):
        create_directories_and_copy_files(
            fold, args.continue_dir, f'random_{i}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare data for GPT-4V evaluation based on CLIP results.')
    parser.add_argument('--clip_model', type=str,
                        choices=['clip-vit-base-patch16', 'clip-vit-base-patch32'], default="clip-vit-base-patch16")
    parser.add_argument('--llava_model', type=str,
                        choices=['llava-v1.5-7b', 'llava-v1.5-13b'], default="llava-v1.5-13b")
    parser.add_argument('--num_rand', type=int, default=180,
                        help='Number of random samples to extract from CLIP results. Default is 180.')
    parser.add_argument('--num_failure', type=int, default=180,
                        help='Number of failure cases to extract from CLIP results. Default is 180.')
    parser.add_argument('--continue_dir', type=str, required=True,
                        help='Directory containing the CLIP sample sets, e.g., "exp_output/2023-11-18-19_56_06".')

    args = parser.parse_args()
    data_info = load_data_info()
    dataset = check_directories_in_list(args.continue_dir)
    prepare_data_for_gpt4v(args)
