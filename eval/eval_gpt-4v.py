import os
import re
import base64
import argparse
import requests
import json
from tqdm import tqdm
from itertools import islice
from utils import setup_logging, analyse_unified_output
from datetime import datetime
from data.random_sampler import gen_sample_json


def get_gpt_response(image_path, prompt, api_key=None):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response


def setup(args):
    # setup output directory and logging
    if args.continue_dir is not None:
        args.output_dir = args.continue_dir
        logger = setup_logging(args.output_dir)
        logger.info('Run an experiment continuing from an existing directory')
    else:
        current_time = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
        args.output_dir = f"{args.output_dir}/{current_time}"
        os.makedirs(args.output_dir)
        logger = setup_logging(args.output_dir)
        logger.info('Start a new experiment')

    # Set up logging
    logger.info(args)
    return logger


def main(args):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found.")

    for each_dataset in args.dataset:
        if args.continue_dir is not None:
            logger.info(f'Load an existing unified_input_{each_dataset}.json')
        else:
            gen_sample_json(dataset=each_dataset, args=args)
        # Load the JSON file
        with open(f'{args.output_dir}/unified_input_{each_dataset}.json', 'r') as f:
            data = json.load(f)
        class_names = data['class_names']
        # setup prompt for the dataset
        prompt = f"""Given the image, answer the following question using the specified format. 
        Question: What’s in this image? Choices: {class_names}. 

        Please respond with the following format:
        ---BEGIN FORMAT TEMPLATE---
        Answer Choice: [Your Answer Choice Here]
        Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1]
        Reasoning: [Your Reasoning Behind This Answer Here]
        ---END FORMAT TEMPLATE---

        Do not deviate from the above format. Repeat the format template for the answer.
        """
        first_flag = True
        for item_id, item in tqdm(data['samples'].items(), desc="Processing Samples"):
            # for item_id, item in tqdm(islice(data['samples'].items(), 5), desc="Processing Samples"):
            unified_output = {}
            image_path = os.path.join(
                args.data_dir, each_dataset, item['image'])
            response = get_gpt_response(
                image_path=image_path, prompt=prompt, api_key=api_key).json()
            response.update(
                {'image_path': image_path})
            # logger.info(f'response: {response}')
            if 'error' in response:
                logger.warning(f'{args.model_name} returns server error')
            else:
                answer_str = response['choices'][0]['message']['content']
                logger.info(f'answer by gpt-4v:\n{answer_str}')
                predicted_class = None
                for class_name in class_names:
                    pattern = re.compile(
                        f'Answer Choice: {class_name}', re.IGNORECASE)
                    if pattern.search(answer_str):
                        # if f'Answer Choice: {class_name}' in answer_str:
                        predicted_class = class_name
                        break
                if not predicted_class:
                    logger.info('query failed')
                    continue
                logger.info(f'predicted_class: {predicted_class}')
                # pred_class_id = dataset.class_to_idx[pred_class_name]
                response.update({'predicted_class': predicted_class})
            # Store unified output jsonl
            unified_output['dataset'] = each_dataset
            unified_output['domain'] = item['domain']
            unified_output['subject'] = item['subject']
            unified_output['true_class'] = item['class']
            unified_output['predicted_class'] = predicted_class
            unified_output['image'] = item['image']
            unified_output['id'] = item_id
            mode = 'w' if first_flag else 'a'
            with open(f'{args.output_dir}/unified_output_{args.model_name}.jsonl', mode) as f:
                f.write(json.dumps(unified_output) + "\n")
            first_flag = False
    analyse_unified_output(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate GPT-4v in distribution shifts')
    parser.add_argument('--data_dir', type=str,
                        default="/home/guanglinzhou/scratch/gpt-4v-distribution-shift")
    parser.add_argument('--dataset', type=str, nargs='+', default=["PACS"])
    parser.add_argument('--output_dir', type=str, default="./exp_output")
    parser.add_argument('--num_sample', type=int, default=20,
                        help="the number of samples for each class")
    parser.add_argument('--model_name', type=str,
                        choices=['gpt-4-vision-preview'], default="gpt-4-vision-preview")
    parser.add_argument('--continue_dir', type=str, default=None,
                        help="evaluate llava on the same sample sets with CLIP, i.e., exp_output/2023-11-18-19_56_06")
    parser.add_argument('--save_samples', action='store_true',
                        help='whether save the sample sets into output_dir')
    args = parser.parse_args()

    logger = setup(args)
    main(args)
