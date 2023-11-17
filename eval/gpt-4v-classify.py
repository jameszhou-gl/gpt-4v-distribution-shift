import os
import re
import base64
import argparse
import requests
import json
import logging
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader

import data.datasets as datasets
import eval.utils as utils


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
    current_time = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    args.output_dir = f"{args.output_dir}/{current_time}"
    os.makedirs(args.output_dir)
    logging.basicConfig(filename=f'{args.output_dir}/log.txt', level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main(args):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found.")
    # prepare dataset
    logging.info(f'set up dataset: {args.dataset}')
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir)
    else:
        raise NotImplementedError
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        out, in_ = utils.split_dataset(env,
                                       int(len(env)*args.holdout_fraction),
                                       utils.seed_hash(args.trial_seed, env_i))
        print(len(out))
        in_splits.append(in_)
        out_splits.append(out)

    def custom_collate(batch):
        return batch
    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=1, collate_fn=custom_collate) for env in out_splits]
    eval_loader_names = ['env{}_out'.format(i)
                         for i in range(len(out_splits))]
    evals = zip(eval_loader_names, eval_loaders)
    # setup prompt for the dataset
    prompt = f"""Given the image, answer the following question using the specified format. Question: What’s in this image? Choices: {dataset.class_names}. 

    Please respond with the following format:
    ---BEGIN FORMAT TEMPLATE---
    Answer Choice: [Your Answer Choice Here]
    Confidence Score: [Your Numerical Prediction Confidence Score Here]
    Reasoning: [Your Reasoning Behind This Answer Here]
    ---END FORMAT TEMPLATE---

    Do not deviate from the above format. Repeat the format template for the answer.
    """

    # prompt = f'Question: What’s in this image?; Multiple Choices: {dataset.class_names}; Output an answer chosen from the choices list, the numerical prediction confidence score for the answer choice and reason'
    logging.info(f'setup prompt: {prompt}')
    results = {}
    for name, loader in evals:
        correct = 0
        total = 0
        for sample in tqdm(loader, desc=f"Processing {name}"):
            img_path, gt_class_id = sample[0][0], sample[0][1]
            logging.info(f'img_path: {img_path}, gt_class_id: {gt_class_id}')
            response = get_gpt_response(
                image_path=img_path, prompt=prompt, api_key=api_key).json()
            response.update(
                {'image_path': img_path, 'gt_class_id': gt_class_id})
            logging.info(f'response: {response}')
            if 'error' in response:
                logging.info('error')
                pass
            else:
                answer_str = response['choices'][0]['message']['content']
                pred_class_name = None
                for class_name in dataset.class_names:
                    pattern = re.compile(
                        f'Answer Choice: {class_name}', re.IGNORECASE)
                    if pattern.search(answer_str):
                        # if f'Answer Choice: {class_name}' in answer_str:
                        pred_class_name = class_name
                        break
                if not pred_class_name:
                    logging.info('query failed')
                    continue
                logging.info(f'pred_class_name: {pred_class_name}')
                pred_class_id = dataset.class_to_idx[pred_class_name]
                response.update({'pred_class_name': pred_class_name,
                                'pred_class_id': pred_class_id})
                correct += 1 if pred_class_id == gt_class_id else 0
                total += 1
            with open(f'{args.output_dir}/output.jsonl', 'a') as file:
                file.write(json.dumps(response) + '\n')
        results[name] = float(correct/total)
    with open(f'{args.output_dir}/results.jsonl', 'w') as f:
        f.write(json.dumps(results, sort_keys=True) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GPT-4v in distribution shift scenarios')
    parser.add_argument('--data_dir', type=str,
                        default="/l/users/guanglin.zhou/gpt-4v-distribution-shift")
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--output_dir', type=str, default="./exp_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.01)
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    args = parser.parse_args()

    setup(args)
    main(args)
