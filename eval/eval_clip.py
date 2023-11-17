import json
import os
import torch
import argparse
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from utils import setup_logging
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


def main(args):
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if not torch.cuda.is_available():
        raise ValueError
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for each_dataset in args.dataset:
        gen_sample_json(dataset=each_dataset, num_sample=20,
                        data_dir=args.data_dir, output_dir=args.output_dir)
        # Load the JSON file
        with open(f'{args.output_dir}/samples_in_{each_dataset}.json', 'r') as f:
            data = json.load(f)
        # Extract class names from the JSON file
        class_names = data['class_names']
        correct = 0
        total = 0
        # Process each image in the JSON file's 'samples' section
        for item_id, item in data['samples'].items():
            image_path = os.path.join(
                args.data_dir, each_dataset, item['image'])

            # Load the image
            image = Image.open(image_path).convert("RGB")

            # Process the image and text with CLIP processor
            inputs = processor(text=class_names, images=image,
                               return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get the true class label from JSON
            true_class = item['class']

            # Process the image with CLIP and get the predicted class label
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                predicted_class_id = probs.argmax().item()
                predicted_class = class_names[predicted_class_id]

            # Compare the predicted class label with the true class
            is_correct = predicted_class == true_class
            correct += 1 if is_correct else 0
            total += 1
            # Store or output the results
            logger.info(
                f"Item ID: {item_id}, True Class: {true_class}, Predicted Class: {predicted_class}, Correct: {is_correct}")
        logger.info('Prediction accuracy: {}'.format(float(correct/total)))
        logger.info("Processing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate CLIP in distribution shifts')
    parser.add_argument('--data_dir', type=str,
                        default="/home/guanglinzhou/scratch/gpt-4v-distribution-shift")
    parser.add_argument('--dataset', type=str, nargs='+', default=["PACS"])
    parser.add_argument('--output_dir', type=str, default="./exp_output")
    args = parser.parse_args()

    logger = setup(args)
    main(args)
