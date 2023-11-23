# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import shutil
import json
import argparse
from collections import defaultdict

from tqdm import tqdm
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset
from wilds.datasets.iwildcam_dataset import IWildCamDataset


def metadata_values(wilds_dataset, metadata_name):
    metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
    metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
    return sorted(list(set(metadata_vals.view(-1).tolist())))

def process_camelyon(data_dir):
    """
    Processes the Camelyon dataset from WILDS and restructures it into a format similar to the PACS and VLCS datasets. 
    This function organizes the raw dataset into a hierarchical directory structure based on domains and classes. 
    The resulting file structure is as follows:
    ├── hospital_0
        ├── class_1
        ├── class_2
        ├── ...
    ├── hospital_1
    └── hospital_2
    Each 'domain' directory represents a distinct subset or category within the dataset. 
    Within each domain directory, there are further subdirectories for each class, where the class-specific data is stored.
    """
    dataset = Camelyon17Dataset(root_dir=data_dir)

    with open('./data/dataset_info.json', 'r') as f:
        data = json.load(f)
        dataset_info = data['camelyon17_v1.0_processed']
    class_name_dict = {str(index): value for index, value in enumerate(dataset_info['class_names'])}
    file_structure = defaultdict(lambda: defaultdict(lambda: []))
    
    metadata_name = "hospital"
    for i, metadata_value in enumerate(
            metadata_values(dataset, metadata_name)):
        domain_name = metadata_name + "_" + str(metadata_value)

        metadata_index = dataset.metadata_fields.index(metadata_name)
        metadata_array = dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]
        
        indices = subset_indices
        for i in indices:
            img_path = os.path.join(
                data_dir,
                'camelyon17_v1.0',
                dataset._input_array[i])
            y = dataset.y_array[i]
            class_name = class_name_dict[str(y.item())]
            file_structure[domain_name][class_name].append(img_path)
    print('file_structure is done')
    # Directory to set up
    base_dir = f'{data_dir}/camelyon17_v1.0_processed'

    # Check if the base directory exists, raise an error if it does
    if os.path.exists(base_dir):
        raise FileExistsError(f"The directory {base_dir} already exists.")
    else:
        os.makedirs(base_dir)

    # Iterate over the file structure
    for domain, classes in file_structure.items():
        domain_dir = os.path.join(base_dir, domain)

        # Create domain directory
        if not os.path.exists(domain_dir):
            os.makedirs(domain_dir)
        # Iterate over each class in the domain
        for class_name, img_paths in tqdm(classes.items(), desc=f"Processing {domain}"):
            class_dir = os.path.join(domain_dir, class_name)
            
            # Create class directory
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Copy each image to the new directory
            for img_path in img_paths:
                shutil.copy(img_path, class_dir)

    print("Files copied successfully to the new structure: camelyon17_v1.0_after_process.")
    
    
def process_fmow(data_dir):
    """
    Processes the Fmow dataset from WILDS and restructures it into a format similar to the PACS and VLCS datasets. 
    This function organizes the raw dataset into a hierarchical directory structure based on domains and classes. 
    The resulting file structure is as follows:
    ├── region_0
        ├── class_1
        ├── class_2
        ├── ...
    ├── region_1
    └── region_2
    Each 'domain' directory represents a distinct subset or category within the dataset. 
    Within each domain directory, there are further subdirectories for each class, where the class-specific data is stored.
    """
    dataset = FMoWDataset(root_dir=data_dir)

    with open('./data/dataset_info.json', 'r') as f:
        data = json.load(f)
        dataset_info = data['fmow_v1.1_processed']
    class_name_dict = {str(index): value for index, value in enumerate(dataset_info['class_names'])}
    file_structure = defaultdict(lambda: defaultdict(lambda: []))
    
    metadata_name = "region"
    for i, metadata_value in enumerate(
            metadata_values(dataset, metadata_name)):
        domain_name = metadata_name + "_" + str(metadata_value)

        metadata_index = dataset.metadata_fields.index(metadata_name)
        metadata_array = dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]
        
        indices = subset_indices
        for i in indices:
            idx = dataset.full_idxs[i]
            img_path = os.path.join(
                data_dir,
                'fmow_v1.1', 'images', f'rgb_img_{idx}.png')
            y = dataset.y_array[i]
            class_name = class_name_dict[str(y.item())]
            file_structure[domain_name][class_name].append(img_path)
    print('file_structure is done')
    # Directory to set up
    base_dir = f'{data_dir}/fmow_v1.1_processed'

    # Check if the base directory exists, raise an error if it does
    if os.path.exists(base_dir):
        raise FileExistsError(f"The directory {base_dir} already exists.")
    else:
        os.makedirs(base_dir)

    # Iterate over the file structure
    for domain, classes in file_structure.items():
        domain_dir = os.path.join(base_dir, domain)

        # Create domain directory
        if not os.path.exists(domain_dir):
            os.makedirs(domain_dir)
        # Iterate over each class in the domain
        for class_name, img_paths in tqdm(classes.items(), desc=f"Processing {domain}"):
            class_dir = os.path.join(domain_dir, class_name)
            
            # Create class directory
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Copy each image to the new directory
            for img_path in img_paths:
                shutil.copy(img_path, class_dir)

    print("Files copied successfully to the new structure: fmow_v1.1_processed.")


def process_iwildcam(data_dir):
    """
    Processes the Iwildcam dataset from WILDS and restructures it into a format similar to the PACS and VLCS datasets. 
    This function organizes the raw dataset into a hierarchical directory structure based on domains and classes. 
    The resulting file structure is as follows:
    ├── location_0
        ├── class_1
        ├── class_2
        ├── ...
    ├── location_2
    ├── ...
    └── location_245
    Each 'domain' directory represents a distinct subset or category within the dataset. 
    Within each domain directory, there are further subdirectories for each class, where the class-specific data is stored.
    """
    dataset = IWildCamDataset(root_dir=data_dir)
    with open('dataset_info.json', 'r') as f:
        data = json.load(f)
        dataset_info = data['iwildcam_v2.0_processed']
    class_name_dict = {str(index): value for index, value in enumerate(dataset_info['class_names'])}
    file_structure = defaultdict(lambda: defaultdict(lambda: []))
    
    metadata_name = "location" #according to the metadata 
    #print(metadata_values(dataset, metadata_name))
    for i, metadata_value in enumerate(
            metadata_values(dataset, metadata_name)):
        domain_name = metadata_name + "_" + str(metadata_value)
    
        metadata_index = dataset.metadata_fields.index(metadata_name)
        metadata_array = dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]
        indices = subset_indices
        for i in indices:
            idx = dataset._input_array[i]
            print(idx)
            img_path = os.path.join(
                data_dir,
                'iwildcam_v2.0', 'train', f'{idx}')
            y = dataset.y_array[i]
            class_name = class_name_dict[str(y.item())]
            file_structure[domain_name][class_name].append(img_path)
    print('file_structure is done')
    # Directory to set up
    base_dir = f'{data_dir}/iwildcam_v2.0_processed'

    # Check if the base directory exists, raise an error if it does
    if os.path.exists(base_dir):
        raise FileExistsError(f"The directory {base_dir} already exists.")
    else:
        os.makedirs(base_dir)

    # Iterate over the file structure
    for domain, classes in file_structure.items():
        domain_dir = os.path.join(base_dir, domain)

        # Create domain directory
        if not os.path.exists(domain_dir):
            os.makedirs(domain_dir)
        # Iterate over each class in the domain
        for class_name, img_paths in tqdm(classes.items(), desc=f"Processing {domain}"):
            class_dir = os.path.join(domain_dir, class_name)
            
            # Create class directory
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Copy each image to the new directory
            for img_path in img_paths:
                shutil.copy(img_path, class_dir)

    print("Files copied successfully to the new structure: iwildcam_v2.0_processed.")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Restructures WILDS datasets into a similar format in PACS, VLCS')
    parser.add_argument('--data_dir', type=str,
                        default="/l/users/zhongyi.han/data")
    args = parser.parse_args()

    process_camelyon(data_dir=args.data_dir)
    process_fmow(data_dir=args.data_dir)
    process_iwildcam(data_dir=args.data_dir)