import os
import re
import argparse
import json
import PIL.Image
from tqdm import tqdm
from evaluation.utils import setup_logging, analyse_unified_output
from datetime import datetime
from data.random_sampler import gen_sample_json
import google.generativeai as genai


def get_gemini_response(image_path, prompt, model_name, api_key=None):
    genai.configure(api_key=api_key)
    img = PIL.Image.open(image_path)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content([prompt, img], stream=True)
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


def search_pred_info(item_id, answer_by_gpt, class_names):
    predicted_class = None
    for class_name in class_names:
        # Pattern to match 'Answer Choice: [class_name]' or 'Answer Choice: class_name' (case-insensitive)
        pattern = re.compile(
            r"Answer Choice:\s*(?:\[)?'?\"?" +
            re.escape(class_name) + r"'?\"?(?:\])?",
            re.IGNORECASE
        )
        if pattern.search(answer_by_gpt):
            predicted_class = class_name
            break
    if not predicted_class:
        logger.warning(
            'Query failed for item {}; Check the answer or the pattern matching carefully!'.format(item_id))
    # Regular expression patterns to extract Confidence Score (0~1) and Reasoning
    confidence_score_pattern = r'Confidence Score:\s*([0-9]*\.?[0-9]+)'
    reasoning_pattern = r'Reasoning:\s*(.+)'

    # Extract Confidence Score
    confidence_score_match = re.search(
        confidence_score_pattern, answer_by_gpt, re.DOTALL)
    if confidence_score_match:
        confidence_score = confidence_score_match.group(1).strip()
    else:
        confidence_score = None

    # Extract Reasoning
    reasoning_match = re.search(reasoning_pattern, answer_by_gpt, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        reasoning = None
    return predicted_class, confidence_score, reasoning


def main(args):
    api_key = args.GOOGLE_API_KEY
    if not api_key:
        raise ValueError("GOOGLE API key not found.")

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
        Question: What is in this image? Choices: {class_names}. 

        Please respond with the following format:
        ---BEGIN FORMAT TEMPLATE---
        Answer Choice: [Your Answer Choice Here]
        Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1]
        Reasoning: [Your Reasoning Behind This Answer Here]
        ---END FORMAT TEMPLATE---

        Do not deviate from the above format. Repeat the format template for the answer.
        """
        output_file_path = os.path.join(
            args.output_dir, f'unified_output_{args.model_name}.jsonl')
        if os.path.exists(output_file_path):
            logger.info(f'File already exists: {output_file_path}')
            first_flag = False
            id_list = []
            with open(output_file_path, 'r') as file:
                for line in file:
                    # Convert each line to a JSON object
                    jsonl_data = json.loads(line)
                    # Extract the 'id' and add it to the list
                    id_list.append(jsonl_data['id'])

        else:
            first_flag = True
            id_list = []
        server_error_num = 0
        total_samples_count = len(data['samples'])
        remaining_samples_count = len(data['samples'])-len(id_list)
        if remaining_samples_count == 0:
            logger.info(
                f'All {total_samples_count} samples in unified_input_{each_dataset}.json have been '
                f'processed. Exiting as there are no more samples to test with unified_output_{args.model_name}.jsonl.'
            )
            # Perform any necessary cleanup or finalization here
            exit()
        else:
            logger.info(
                f'Continuing testing: {remaining_samples_count} samples remaining.')

        for item_id, item in tqdm(data['samples'].items(), desc="Processing Samples"):
            # for item_id, item in tqdm(islice(data['samples'].items(), 5), desc="Processing Samples"):
            if item_id in id_list:
                continue
            unified_output = {}
            if args.continue_dir is not None:
                image_path = os.path.join(
                    args.continue_dir, item['image'])
            else:
                image_path = os.path.join(
                    args.data_dir, each_dataset, item['image'])
            try:
                response = get_gemini_response(
                    image_path=image_path, prompt=prompt, model_name=args.model_name, api_key=api_key)
                response.resolve()
                answer_by_gemini = response.text
                logger.info(f'answer by gemini:\n{answer_by_gemini}')
                predicted_class, confidence_score, reasoning = search_pred_info(
                    item_id, answer_by_gemini, class_names)
                if not predicted_class:
                    continue
                logger.info(f'predicted_class: {predicted_class}')
                # Store unified output jsonl
                unified_output['dataset'] = each_dataset
                unified_output['domain'] = item['domain']
                unified_output['subject'] = item['subject']
                unified_output['true_class'] = item['class']
                unified_output['predicted_class'] = predicted_class
                unified_output['confidence_score'] = confidence_score
                unified_output['reasoning'] = reasoning
                unified_output['image'] = item['image']
                unified_output['id'] = item_id
                mode = 'w' if first_flag else 'a'
                with open(f'{args.output_dir}/unified_output_{args.model_name}.jsonl', mode) as f:
                    f.write(json.dumps(unified_output) + "\n")
                first_flag = False
            except Exception as e:
                logger.warning(f'{args.model_name} returns server error')
                logger.warning(f'{type(e).__name__}: {e}')
                server_error_num += 1
                if server_error_num == 20:
                    logger.warning(
                        'The server error reaches 20 times, terminating the code')
                    exit()
    analyse_unified_output(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Gemini in distribution shifts')
    parser.add_argument('--data_dir', type=str,
                        default="/home/guanglinzhou/scratch/gpt-4v-distribution-shift")
    parser.add_argument('--dataset', type=str, nargs='+', default=["PACS"])
    parser.add_argument('--output_dir', type=str, default="./exp_output")
    parser.add_argument('--num_sample', type=int, default=20,
                        help="the number of samples for each class")
    parser.add_argument('--model_name', type=str,
                        choices=['gemini-pro-vision'], default="gemini-pro-vision")
    parser.add_argument('--continue_dir', type=str, default=None,
                        help="evaluate gpt on the same sample sets with CLIP, i.e., exp_output/2023-11-18-19_56_06")
    parser.add_argument('--save_samples', action='store_true',
                        help='whether save the sample sets into output_dir')
    parser.add_argument('--GOOGLE_API_KEY', type=str, required=True)

    args = parser.parse_args()

    logger = setup(args)
    main(args)
