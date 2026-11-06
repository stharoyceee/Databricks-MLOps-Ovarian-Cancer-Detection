from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Class to keep track of both training and validation files 
class OvarianDataset(Dataset):
    def __init__(self, file_paths, targets, transform):
        self.file_paths = file_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    # Normalising all images to 244,244 size and making it ready to feed into a pretrained CNN
inference_transforms = transforms.Compose([
    transforms.Resize((244,244)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])