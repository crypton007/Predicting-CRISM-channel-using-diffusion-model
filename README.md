# Predicting spectral channels related to mineralogy from CRISM spectral bands using deep learning

## Introduction

The specifc objective of this thesis is to take images from CRISM, chop the special information down to the 4 bands, that we fnd in CaSSIS as well, and predict spectral channels related to mineralogy (which are available in the original CRISM data) thus enhancing the predictive capabilities of Martian mineralogical signatures

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Dataset Generation](#Dataset Generation)
4. [Model Training](#Model Training)
5. [Inference](#Inference)

## Getting Started

To train a stable diffusion model we need the required dataset and powerful GPU as well as sufficient hard disk storage.
The code is divided into two environments, the data generation part is done using JupyterHub and the model training part is done on the workstation with high GPU resource.

## installation

To install and set up this project, follow these steps:

1. **Clone the repository**

   First, clone the repository to your local machine using Git.

   ```sh
   git clone git@mygit.th-deg.de:PANDAS/sahani.git
    ```
2. **Install the required dependencies**
    ```sh
    pip install -r requirements.txt
    ``` 
    Make sure that the torch version matches with the CUDA


## Dataset Generation

The Data can be generated using the Dataset_preparation.ipynb file in JupytherHub folder, simply provide the correct root directory and run the cell. This will also generate the required metadata.
Each process is defined in appropriate cells

Unzip the dataset and split them into train and test folders using your preferred method.

Run the export_dataset.py providing appropriate file path of train samples to create a .parquet file to be used by the model.  

## Model Training

1. **Unet**
    
Simply provide the path of the train samples and run the unet_training.py

2. **Instruct pix2pix**
    
Fine-tuning instructPix2Pix for our task provide appropriate arguments and then run the finetune_instruct_Pix2Pix_final.py

```sh
export DATASET_ID="dataset_train/data_mapping"
export OUTPUT_DIR="crism_train_sd"
export MODEL_ID="timbrooks/instruct-pix2pix"

accelerate launch --mixed_precision="fp16" finetune_instructpix2pix_final.py \
--pretrained_model_name_or_path=$MODEL_ID \
--dataset_name=$DATASET_ID \
--use_ema \
--enable_xformers_memory_efficient_attention \
--resolution=512 \
--train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing \
--max_train_steps=500 \
--learning_rate=5e-05 --lr_warmup_steps=0 \
--mixed_precision=fp16 \
--val_image_url="/home/as05412/amanrepos/crism_sd/dataset_newChannel/crism_dataset_newChannel/test/frt0000a91c/original_image.png" \
--validation_prompt="predict carbonates" \
--seed=42 \
--output_dir=$OUTPUT_DIR
``` 
## Inference

Go throught the final_inference.ipynb notebook and follow the code to create appropriate metric scores.
