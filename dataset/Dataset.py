import torch
import torchvision.transforms as transforms
from PIL import Image
from glob import glob

class dataset(torch.utils.data.Dataset):

    def __init__(self, root_A, root_B, opt):
        # self.images_A = glob(root_A + "/*.jpg")
        # self.images_B = glob(root_B + "/*.jpg")

        self.images_A = glob(root_A + "/*.png")
        self.images_B = glob(root_B + "/*.png")
        assert len(self.images_A) != 0
        assert len(self.images_B) != 0
        self.transfroms = transforms.Compose([
            transforms.Resize(opt.loadsize),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(opt.cropsize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __getitem__(self, index):
        img_A = self.images_A[index]
        img_A = Image.open(img_A).convert('RGB')
        img_A = self.transfroms(img_A)
        img_B = self.images_B[index]
        img_B = Image.open(img_B).convert('RGB')
        img_B = self.transfroms(img_B)
        return img_A, img_B
    
    def __len__(self):
        return min(len(self.images_A), len(self.images_B))
