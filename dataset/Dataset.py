import torch
import torchvision.transforms as transforms
from PIL import Image
from glob import glob

class dataset(torch.utils.data.Dataset):

    def __init__(self, root, opt):
        self.images = glob(root + "/*.jpg")
        assert len(self.images) != 0
        self.transfroms = transforms.Compose([
            transforms.Resize(opt.load_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(opt.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.images)