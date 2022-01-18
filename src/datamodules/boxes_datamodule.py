import torch
import random
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from torch.utils.data import random_split
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

boxes_dataset_mean = torch.tensor([0.3150, 0.3402, 0.3735])
boxes_dataset_std =  torch.tensor([0.2519, 0.2499, 0.2480])

class BoxesDataset(Dataset):
    def __init__(self, image_dir : str, class_limit : int = 1000, per_class_limit :int = 1000, transforms = None):
        self.image_dir = image_dir
        self.class_limit = class_limit
        self.per_class_limit = per_class_limit
        self.transforms = transforms
        self._init_dataset()

    def _init_dataset(self):

        image_names = os.listdir(self.image_dir)
        image_labels = [int(os.path.splitext(img_name)[0].split("_")[1]) for img_name in image_names]
        img_classes = list(set(image_labels))
        num_classes = len(img_classes)
        images_by_class = [[] for x in range(num_classes)]
        for index, label in enumerate(image_labels):
            images_by_class[label].append(image_names[index])

        if self.class_limit < len(img_classes):
            images_by_class =  random.sample(images_by_class, self.class_limit)
        for index, img_class in enumerate(images_by_class):
            if len(img_class) > self.per_class_limit:
                images_by_class[index] = random.sample(img_class, self.per_class_limit)
        self.image_names = []
        self.labels = []
        for class_index, img_class in enumerate(images_by_class):
            self.labels += [class_index] * len(img_class)
            self.image_names += img_class
    
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_dir, self.image_names[index]))
        if self.transforms != None:
            img = self.transforms(img)
        else:
            img = transforms.ToTensor()(img)
        return img, self.labels[index]
    
class BoxesDataModule(pl.LightningDataModule):
    def __init__(self, image_dir, batch_size, class_limit = 1000, per_class_limit = 1000, num_workers = 8,
                 train_transforms = None, val_transforms = None, val_fraction = 0.2):
        self.image_dir = image_dir
        self.class_limit = class_limit
        self.per_class_limit = per_class_limit
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.seed = 42
        self.val_fraction = val_fraction

        dataset = BoxesDataset(self.image_dir, self.class_limit, self.per_class_limit)
        len_train = int(len(dataset) * (1 - self.val_fraction))
        len_val = len(dataset) - len_train
        self.num_samples = len_train 
        self.num_classes = 456
    
    def train_dataloader(self):
        transforms = self._default_transforms() if self.train_transforms is None else self.train_transforms
        dataset = BoxesDataset(self.image_dir, self.class_limit, self.per_class_limit, transforms=transforms)
        len_train = int(len(dataset) * (1 - self.val_fraction))
        len_val = len(dataset) - len_train
        train_dataset, _ = random_split(dataset, (len_train, len_val), torch.Generator().manual_seed(self.seed))
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms
        dataset = BoxesDataset(self.image_dir, self.class_limit, self.per_class_limit, transforms=transforms)
        len_train = int(len(dataset) * (1 - self.val_fraction))
        len_val = len(dataset) - len_train
        _, val_dataset = random_split(dataset, (len_train, len_val), torch.Generator().manual_seed(self.seed))
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def _default_transforms(self):
        return transforms.Compose([transforms.ToTensor(), self.boxes_normalization()])
    
    @staticmethod
    #normalization value for full dataset (2/12/2021) ~ 40k images
    def boxes_normalization():
        return transforms.Normalize(mean=(0.3150, 0.3402, 0.3735), std=(0.2519, 0.2499, 0.2480))

if __name__=="__main__":
    dataset = BoxesDataset("C:/Users/VRPC/clustering/data/images/images", per_class_limit=50, class_limit=10)
    im = dataset[0]
    print("done")