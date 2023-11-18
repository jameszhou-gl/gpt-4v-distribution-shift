import os
import random
import json
import logging
import shutil


def save_samples(data, args):
    os.makedirs(args.output_dir, exist_ok=True)
    # Iterate over the samples and copy each image
    for sample in data['samples'].values():
        image_relative_path = sample['image']
        source_image_path = os.path.join(
            args.data_dir, data['dataset'], image_relative_path)
        destination_image_path = os.path.join(
            args.output_dir, data['dataset'], image_relative_path)
        # Create subdirectories in the new data directory if they don't exist
        os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
        # Copy the image
        shutil.copy(source_image_path, destination_image_path)


def gen_sample_json(dataset='PACS', args=None):
    logger = logging.getLogger('gpt-4v-distribution-shift-logger')
    logger.info(
        f'sampling data examples in {dataset}, and writing into unified_input_{dataset}.json')
    # Load the JSON file
    with open('./data/dataset_info.json', 'r') as f:
        data = json.load(f)
        if dataset not in data:
            raise ValueError(
                f"Dataset '{dataset}' not supported! Please update metadata for such dataset in dataset_info.json")
        dataset_info = data[dataset]
    domains = dataset_info['domains']
    class_names = dataset_info['class_names']
    subject = dataset_info['subject']
    logger.info('Processing {}: {} domains, {} classes'.format(
        dataset, len(domains), len(class_names)))
    selected_images_info = {'dataset': dataset, 'domains': domains,
                            'class_names': class_names, 'samples': {}}

    # Iterate through each domain and class
    for domain in domains:
        domain_path = os.path.join(args.data_dir, dataset, domain)
        if os.path.exists(domain_path) and os.path.isdir(domain_path):
            for class_id, class_name in enumerate(selected_images_info['class_names']):
                class_path = os.path.join(domain_path, class_name)
                images = [os.path.join(domain, class_name, img) for img in os.listdir(
                    class_path) if img.endswith((".jpg", ".png"))]

                # Randomly sample args.num_sample images
                if len(images) >= args.num_sample:
                    sampled_images = random.sample(images, args.num_sample)
                    for image in sampled_images:
                        image_id = len(selected_images_info['samples']) + 1
                        selected_images_info['samples'][str(image_id)] = {
                            "domain": domain,
                            "class": class_name,
                            "image": image,
                            "class_id": str(class_id),
                            "subject": subject
                        }
                else:
                    logger.info(
                        f"Not enough images in {os.path.join(domain, class_name)} to sample {args.num_sample} images.")

    # Write selected images information to a JSON file
    with open(f'{args.output_dir}/unified_input_{dataset}.json', 'w') as f:
        json.dump(selected_images_info, f, indent=4)
    if args.save_samples:
        save_samples(data=selected_images_info, args=args)
        logger.info(f'save the sampled data into {args.output_dir}')
