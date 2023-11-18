import os
import random
import json
import logging


def gen_sample_json(dataset='PACS', num_sample=20, data_dir=None, output_dir=None):
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
        domain_path = os.path.join(data_dir, dataset, domain)
        if os.path.exists(domain_path) and os.path.isdir(domain_path):
            for class_id, class_name in enumerate(selected_images_info['class_names']):
                class_path = os.path.join(domain_path, class_name)
                images = [os.path.join(domain, class_name, img) for img in os.listdir(
                    class_path) if img.endswith((".jpg", ".png"))]

                # Randomly sample num_sample images
                if len(images) >= num_sample:
                    sampled_images = random.sample(images, num_sample)
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
                        f"Not enough images in {os.path.join(domain, class_name)} to sample {num_sample} images.")

    # Write selected images information to a JSON file
    with open(f'{output_dir}/unified_input_{dataset}.json', 'w') as f:
        json.dump(selected_images_info, f, indent=4)
