import json
import os
import torch
import argparse
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from evaluation.utils import setup_logging, analyse_unified_output
from data.random_sampler import gen_sample_json


def setup(args):
    # setup output directory and logging
    if args.output_dir[-6:] in 'exp_output':
        current_time = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
        args.output_dir = f"{args.output_dir}/{current_time}"
        os.makedirs(args.output_dir)
    logger = setup_logging(args.output_dir)
    logger.info(args)
    return logger


def main(args):
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained(f'openai/{args.model_name}')
    processor = CLIPProcessor.from_pretrained(f'openai/{args.model_name}')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    gen_sample_json(dataset=args.dataset, args=args)
    # Load the JSON file
    with open(f'{args.output_dir}/unified_input_{args.dataset}.json', 'r') as f:
        data = json.load(f)
    # Extract class names from the JSON file
    class_names = data['class_names']

    for item_id, item in tqdm(data['samples'].items(), desc="Processing Samples"):
        unified_output = {}
        image_path = os.path.join(
            args.data_dir, args.dataset, item['image'])
        # Load the image
        image = Image.open(image_path).convert("RGB")
        # Process the image and text with CLIP processor
        inputs = processor(text=class_names, images=image,
                           return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Process the image with CLIP and get the predicted class label
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            predicted_class_id = probs.argmax().item()
            predicted_class = class_names[predicted_class_id]

        # Store unified output jsonl
        unified_output['dataset'] = args.dataset
        unified_output['domain'] = item['domain']
        unified_output['subject'] = item['subject']
        unified_output['true_class'] = item['class']
        unified_output['predicted_class'] = predicted_class
        unified_output['image'] = item['image']
        unified_output['id'] = item_id
        with open(f'{args.output_dir}/unified_output_{args.model_name}.jsonl', 'a') as jsonl_file:
            jsonl_file.write(json.dumps(unified_output) + '\n')

    logger.info("CLIPModel processes completed.")
    analyse_unified_output(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate CLIP in distribution shifts')
    parser.add_argument('--data_dir', type=str,
                        default="/home/guanglinzhou/scratch/gpt-4v-distribution-shift")
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--output_dir', type=str, default="./exp_output")
    parser.add_argument('--num_sample', type=int, default=20,
                        help="the number of samples for each class")
    parser.add_argument('--model_name', type=str,
                        choices=['clip-vit-base-patch16', 'clip-vit-base-patch32'], default="clip-vit-base-patch16")
    parser.add_argument('--save_samples', action='store_true',
                        help='whether save the sample sets into output_dir')
    args = parser.parse_args()

    logger = setup(args)
    main(args)
