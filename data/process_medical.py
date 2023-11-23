# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import shutil
import pandas as pd

from tqdm import tqdm

import argparse

import os
import shutil
import pandas as pd

# Function to restructure the dataset
def restructure_HAM10000(images_base_path, metadata_path):
    
    abbreviations_class = {
    "akiec": "actinic keratoses and intraepithelial carcinoma",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesions",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevi",
    "vasc": "vascular lesions"
    }

    metadata = pd.read_csv(metadata_path)
    # Iterate over each entry in the metadata
    for index, row in metadata.iterrows():
        # Extract class name and dataset name
        class_name = row['dx']
        dataset_name = row['dataset']
        image_filename = row['image_id'] + '.jpg'

        class_name=abbreviations_class[class_name]
        # Create new directory path
        new_directory_path = os.path.join(images_base_path, dataset_name, class_name)
        
        # Create the directory if it does not exist
        if not os.path.exists(new_directory_path):
            os.makedirs(new_directory_path)
        
        # Define the source and destination paths for the image
        src_image_path = os.path.join(images_base_path, image_filename)
        dest_image_path = os.path.join(new_directory_path, image_filename)
        
        # Move the image to the new directory
        if os.path.isfile(src_image_path):  # Check if the source file exists
            shutil.move(src_image_path, dest_image_path)
        else:
            print(f"File {src_image_path} not found.")
    
    print("Dataset restructuring complete.")

# Function to restructure the dataset
def restructure_NIH_Chest_X_ray_14(images_base_path, metadata_path):
    metadata = pd.read_csv(metadata_path)
    for index, row in tqdm(metadata.iterrows()):
        # Get the first finding from the 'Finding Labels' column
        class_name = row['Finding Labels'].split('|')[0]
        dataset_name = row['View Position']
        image_filename = row['Image Index']
        
        # Create new directory path
        images_new_base_path = '/l/users/zhongyi.han/data/NIH_Chest_X_ray_14_processed'
        new_directory_path = os.path.join(images_new_base_path, dataset_name, class_name)
        
        # Create the directory if it does not exist
        if not os.path.exists(new_directory_path):
            os.makedirs(new_directory_path)
        
        # Define the source and destination paths for the image
        src_image_path = os.path.join(images_base_path, image_filename)
        dest_image_path = os.path.join(new_directory_path, image_filename)
        
        # Move the image to the new directory
        if os.path.isfile(src_image_path):  # Check if the source file exists
            shutil.copy(src_image_path, dest_image_path)
        else:
            print(f"File {src_image_path} not found.")
    
    print("Dataset restructuring complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Restructures medical datasets into a similar format in PACS, VLCS')
    parser.add_argument('--data_dir', type=str,
                        default="/l/users/zhongyi.han/data/images-224/images-224")
    parser.add_argument('--metadata_path', type=str,
                        default="/l/users/zhongyi.han/data/Data_Entry_2017.csv")
    args = parser.parse_args()
    restructure_NIH_Chest_X_ray_14(images_base_path=args.data_dir, metadata_path=args.metadata_path)
