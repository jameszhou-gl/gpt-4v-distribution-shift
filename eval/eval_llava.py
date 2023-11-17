import subprocess
import os
import sys
import json
import shutil
import argparse
from utils import setup_logging
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


def convert_samples_question_format_llava(dataset, data, args):
    question_file = f'{args.output_dir}/llava_question_file_{dataset}.jsonl'
    first_flag = True
    class_names = data['class_names']
    for item_id, item in data['samples'].items():
        example_vqa_format = {}
        example_vqa_format['image'] = os.path.join(dataset, item['image'])
        prompt = f"""Given the image, answer the following question using the specified format. Question: What is in this image? Choices: {class_names}. 
        Please respond with the following format:
        ---BEGIN FORMAT TEMPLATE---
        Answer Choice: [Your Answer Choice Here]
        Confidence Score: [Your Numerical Prediction Confidence Score Here]
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


def main(args):
    for each_dataset in args.dataset:
        logger.info(
            f'sampling data examples in {each_dataset}, and writing into json file')
        gen_sample_json(dataset=each_dataset, num_sample=args.num_sample,
                        data_dir=args.data_dir, output_dir=args.output_dir)
        # Load the JSON file
        with open(f'{args.output_dir}/samples_in_{each_dataset}.json', 'r') as f:
            data = json.load(f)
        logger.info(
            'convert the unified sampling format, into the vqa format in LLaVA')
        question_file = convert_samples_question_format_llava(
            each_dataset, data, args)
        logger.info('saving questions in llava vqa format')
        llava_model_vqa = [
            "python", "-m", "llava.eval.model_vqa",
            "--model-path", f"liuhaotian/llava-v1.5-{args.model_size}",
            "--question-file", question_file,
            "--image-folder", f"{args.data_dir}",
            "--answers-file", f"{args.output_dir}/llava-v1.5-{args.model_size}_{each_dataset}_answer.jsonl",
            "--temperature", "0",
            "--conv-mode", "vicuna_v1"
        ]
        subprocess.run(llava_model_vqa)


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
    args = parser.parse_args()

    logger = setup(args)
    main(args)
