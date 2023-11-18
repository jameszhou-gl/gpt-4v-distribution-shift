import subprocess
import re
import os
import sys
import json
import shutil
import argparse
from utils import setup_logging, analyse_unified_output
from datetime import datetime
from data.random_sampler import gen_sample_json


def setup(args):
    # setup output directory and logging
    current_time = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    args.output_dir = f"{args.output_dir}/{current_time}"
    os.makedirs(args.output_dir)
    # Set up logging
    logger = setup_logging(args.output_dir)
    logger.info(args)
    return logger


def convert_unified_input_into_llava_vqa(dataset, data, args):
    question_file = f'{args.output_dir}/unified_input_{dataset}_in_llava_vqa.jsonl'
    first_flag = True
    class_names = data['class_names']
    for item_id, item in data['samples'].items():
        example_vqa_format = {}
        example_vqa_format['image'] = os.path.join(dataset, item['image'])
        prompt = f"""Given the image, answer the following question using the specified format. Question: What is in this image? Choices: {class_names}. 
        Please respond with the following format:
        ---BEGIN FORMAT TEMPLATE---
        Answer Choice: [Your Answer Choice Here]
        Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1]
        Reasoning: [Your Reasoning Behind This Answer Here]
        ---END FORMAT TEMPLATE---

        Do not deviate from the above format. Repeat the format template for the answer.
        """
        example_vqa_format['text'] = prompt
        example_vqa_format['subject'] = item['subject']
        example_vqa_format['question_id'] = item_id
        mode = 'w' if first_flag else 'a'
        with open(question_file, mode) as jsonl_file:
            jsonl_file.write(json.dumps(example_vqa_format) + '\n')
        first_flag = False
    return question_file


def search_pred_info(answer_by_llava, class_names):
    text_in_llava = answer_by_llava['text']
    predicted_class = None
    for class_name in class_names:
        # Pattern to match 'Answer Choice: [class_name]' or 'Answer Choice: class_name' (case-insensitive)
        pattern = re.compile(
            r"Answer Choice:\s*(?:\[)?'?\"?" +
            re.escape(class_name) + r"'?\"?(?:\])?",
            re.IGNORECASE
        )
        if pattern.search(text_in_llava):
            predicted_class = class_name
            break
    if not predicted_class:
        raise ValueError('Query failed for item {}; Check the answer or the pattern matching carefully!'.format(
            answer_by_llava['question_id']))
    # Regular expression patterns to extract Confidence Score (0~1) and Reasoning
    confidence_score_pattern = r'Confidence Score:\s*([0-9]*\.?[0-9]+)'
    reasoning_pattern = r'Reasoning:\s*(.+)'

    # Extract Confidence Score
    confidence_score_match = re.search(
        confidence_score_pattern, text_in_llava, re.DOTALL)
    if confidence_score_match:
        confidence_score = confidence_score_match.group(1).strip()
    else:
        confidence_score = None

    # Extract Reasoning
    reasoning_match = re.search(reasoning_pattern, text_in_llava, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        reasoning = None
    return predicted_class, confidence_score, reasoning


def convert_llava_answer_into_unified_output(dataset, answer_file, unified_input):
    with open(answer_file, 'r') as file:
        class_names = unified_input['class_names']
        for line in file:
            answer_by_llava = json.loads(line)
            item_id = answer_by_llava['question_id']
            item = unified_input['samples'][item_id]
            unified_output = {}
            # Store unified output jsonl
            unified_output['dataset'] = dataset
            unified_output['domain'] = item['domain']
            unified_output['subject'] = item['subject']
            unified_output['true_class'] = item['class']
            predicted_class, confidence_score, reasoning = search_pred_info(
                answer_by_llava, class_names)
            unified_output['predicted_class'] = predicted_class
            unified_output['image'] = item['image']
            unified_output['id'] = item_id
            unified_output['confidence_score'] = confidence_score
            unified_output['reasoning'] = reasoning
            with open(f'{args.output_dir}/unified_output.jsonl', 'a') as jsonl_file:
                jsonl_file.write(json.dumps(unified_output) + '\n')


def main(args):
    for each_dataset in args.dataset:
        gen_sample_json(dataset=each_dataset, num_sample=args.num_sample,
                        data_dir=args.data_dir, output_dir=args.output_dir)
        # Load the JSON file
        with open(f'{args.output_dir}/unified_input_{each_dataset}.json', 'r') as f:
            data = json.load(f)
        logger.info(
            'convert the unified input format, into llava vqa format')
        question_file = convert_unified_input_into_llava_vqa(
            each_dataset, data, args)
        answer_file = f"{args.output_dir}/output_{each_dataset}_in_llava_vqa.jsonl"
        llava_model_vqa = [
            "python", "-m", "llava.eval.model_vqa",
            "--model-path", f"liuhaotian/llava-v1.5-{args.model_size}",
            "--question-file", question_file,
            "--image-folder", f"{args.data_dir}",
            "--answers-file", answer_file,
            "--temperature", "0",
            "--conv-mode", "vicuna_v1"
        ]
        # Run the subprocess and capture the output
        result = subprocess.run(
            llava_model_vqa, stdout=subprocess.PIPE, text=True)
        # Log stdout
        if result.stdout:
            logger.info("LLaVA Output:\n" + result.stdout)
        convert_llava_answer_into_unified_output(
            dataset=each_dataset, answer_file=answer_file, unified_input=data)
    analyse_unified_output(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate LLaVA in distribution shifts')
    parser.add_argument('--data_dir', type=str,
                        default="/home/guanglinzhou/scratch/gpt-4v-distribution-shift")
    parser.add_argument('--dataset', type=str, nargs='+', default=["PACS"])
    parser.add_argument('--output_dir', type=str, default="./exp_output")
    parser.add_argument('--num_sample', type=int, default=20,
                        help="the number of samples for each class")
    parser.add_argument('--model_size', type=str,
                        choices=['7b', '13b'], default="13b")
    # parser.add_argument('--continue', type=str, default="./exp_output")
    args = parser.parse_args()

    logger = setup(args)
    main(args)
