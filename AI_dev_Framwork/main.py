import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms 
# Imports handle file operations, image loading, and PyTorch utilities

class CustomDataset(Dataset):
    # CustomDataset inherits from PyTorch Dataset
    # Defines how data is loaded and accessed
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.labels = self._load_labels(label_file)
        self.image_filenames = sorted(os.listdir(image_dir))
        self.transform = transform
# Initializes dataset with image paths and labels
# Ensures filenames and labels align
    def _load_labels(self, label_file):
        with open(label_file, 'r') as f:
            labels = [int(line.strip()) for line in f]
        return labels
# Reads labels from file
# Converts them to integers
    def __len__(self):
        return len(self.image_filenames)
# Returns the total number of samples in the dataset
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
# Loads an image and its corresponding label based on the index

if __name__ == "__main__":
    # Main block to set up dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
# Defines transformations to apply to the images (resizing and converting to tensor)
    image_dir = "path/to/images"
    label_file = "path/to/labels.txt"
# Specifies paths to the image directory and label file
    dataset = CustomDataset(image_dir=image_dir, label_file=label_file, transform=transform)
    # Creates an instance of the CustomDataset with the specified paths and transformations
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)