import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
from torchvision import transforms
from segmentation_models_pytorch import Unet
from PIL import Image
from zipfile import ZipFile 
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import segmentation_models_pytorch as sm

# Define the path to the root folder containing multiple subfolders
root_folder = 'dataset_pca/crism_dataset/train'

# Initialize lists to store processed images
processed_feature_images = []
processed_target_images = []

# Assuming transform is defined somewhere in your code
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add other transformations as needed
])

# Loop through all subfolders in the root folder
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)

    # Check if the folder contains both original and edited images
    if 'original_image.png' in os.listdir(folder_path) and 'edited_image.png' in os.listdir(folder_path):
        # Read original and edited images
        feature_image_path = os.path.join(folder_path, 'original_image.png')
        target_image_path = os.path.join(folder_path, 'edited_image.png')

        feature_image = cv2.imread(feature_image_path)
        target_image = cv2.imread(target_image_path)

        if feature_image is not None and target_image is not None:
            # Apply the transformation to the images
            feature_image = transform(feature_image)
            target_image = transform(target_image)

            # Add an extra dimension to the images
            feature_image = feature_image.unsqueeze(0)  # Assuming single image
            target_image = target_image.unsqueeze(0)

            processed_feature_images.append(feature_image)
            processed_target_images.append(target_image)

# Concatenate the images
concatenated_feature_images = torch.cat(processed_feature_images, dim=0)
concatenated_target_images = torch.cat(processed_target_images, dim=0)

# Create a PyTorch Dataset
dataset = TensorDataset(concatenated_feature_images, concatenated_target_images)

# Create a PyTorch DataLoader
batch_size = 8
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=3, activation=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 800

# Calculate total number of iterations
total_iterations = num_epochs * len(data_loader)

# Create a progress bar for the entire training process
progress_bar = tqdm(total=total_iterations, desc="Training", leave=False, disable=False)
losses = []

for epoch in range(num_epochs):
    # Initialize the total loss for this epoch
    total_loss = 0.0

    for feature_image, target_image in data_loader:
        feature_image = feature_image.to(device)
        target_image = target_image.to(device)

        optimizer.zero_grad()
        outputs = model(feature_image)
        loss = criterion(outputs, target_image)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update the progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({"epoch": epoch, "loss": total_loss})

    # Calculate the average loss for this epoch
    average_loss = total_loss / len(data_loader)
    losses.append(average_loss)

    # Print the average loss for this epoch
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, average_loss))

# Close the progress bar
progress_bar.close()

# Save the model
torch.save(model, 'unet_pca_800.pth')

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Loss per epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Save the plot
plt.savefig('loss_plot_pca.png')

# Show the plot
plt.show()
