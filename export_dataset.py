import os
from typing import List

import numpy as np
from datasets import Dataset, Features
from datasets import Image as ImageFeature
from datasets import Sequence
from datasets import Value
from zipfile import ZipFile 
import pandas as pd


def load_prompt(prompt_path: str) -> List[str]:
    with open(prompt_path, "r") as f:
        prompt = f.readlines()
    prompt = [i.strip() for i in prompt]
    return prompt

def generate_examples_simplePrompt(data_paths: List[str],instructions: List[str]):
    def fn():
        for data_path in data_paths:
            yield {
                "original_image": {"path": data_path[0]},
                "edit_prompt": np.random.choice(instructions),
                "cartoonized_image": {"path": data_path[1]},
            }

    return fn

def generate_examples(data_paths: List[str]):
    def fn():
        for data_path in data_paths:
            folder_name =  os.path.basename(os.path.dirname(data_path[0]))
            folder_name = folder_name[:11]
            
            metadata = pd.read_csv('metadata.csv')
            
            # print('name ',metadata['Name'][0])
            print('folder_name ',folder_name)
            # Find matching row in metadata
            matching_row = metadata[metadata['Name'] == (folder_name)]
            print('matching row ',matching_row)
            edit_prompt = f"Longitude: {matching_row['Longitude'].values[0]}, Latitude: {matching_row['Latitude'].values[0]}, Location Name: {matching_row['Location_name'].values[0]}, Predict Green Carbonates if present"

            yield {
                "original_image": {"path": data_path[0]},
                "edit_prompt": edit_prompt,
                "edited_image": {"path": data_path[1]},
            }

    return fn

def main():
    #prompt_path = 'prompts.txt'
    # data_zip = 'SampleDataset_pca.zip'
    data_root = '../dataset_newChannel/crism_dataset_newChannel/train'
    # os.makedirs(data_root, exist_ok=True)
    # with ZipFile(data_zip, 'r') as zip_ref:
    #     zip_ref.extractall(data_root)
    
    #prompts = load_prompt(prompt_path)

    data_paths = os.listdir(data_root)
    data_paths = [os.path.join(data_root, d) for d in data_paths]
    new_data_paths = []
    dataset_path = 'dataset_newChannel/crism_dataset_newChannel/train'

    for data_path in data_paths:
        relative_path = os.path.relpath(data_path, data_root)  # Get the relative path
        original_image = os.path.join(dataset_path,relative_path, "original_image.png")
        edited_image = os.path.join(dataset_path,relative_path, "edited_image.png")
        new_data_paths.append((original_image, edited_image))
        
    # for data_path in data_paths:
    # original_image = os.path.join(data_path, "original_image.png")
    # cartoonized_image = os.path.join(data_path, "cartoonized_image.png")
    # new_data_paths.append((original_image, cartoonized_image))

    generation_fn = generate_examples(new_data_paths)
    print("Creating dataset...")
    ds = Dataset.from_generator(
        generation_fn,
        features=Features(
            original_image=ImageFeature(),
            edit_prompt=Value("string"),
            edited_image=ImageFeature(),
        ),
    )
    # ds = Dataset.from_generator(
    #     generation_fn
    # )

    #ds.save_to_disk('toon_test_data/')
    ds.to_parquet('../dataset_newChannel/crism_dataset_newChannel/data_mapping/crismDataset.parquet')
    
if __name__ == "__main__":
    main()
