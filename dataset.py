import torch

from torchvision.io import read_image, decode_image, ImageReadMode
from torchvision.transforms import Resize
import os
class PneumoniaDataset(torch.utils.data.Dataset):


    
    def __init__(self, dataset_dir_path):
        super().__init__()
        tmp_images = []
        tmp_classes = []

        class_counter = 0
        resize = Resize((128,128))
        for class_dir in os.listdir(dataset_dir_path):
            class_dir_path = os.path.join(dataset_dir_path,class_dir)
            if os.path.isdir(class_dir_path):
                for filename in os.listdir(class_dir_path):
                   
                    target_image = resize(read_image((os.path.join(class_dir_path, filename)), ImageReadMode.GRAY))
              
                   
                    target_image = target_image.float()
                    target_image/=255.0
                    
                    tmp_images.append(target_image)
                    tmp_classes.append(class_counter)
                class_counter+=1
        
        self.images = tmp_images
        self.classes = torch.Tensor(tmp_classes).float()
        self.classes = self.classes.view(self.classes.size(0), 1)
        
        
    def __getitem__(self, index):
        return (self.images[index], self.classes[index])
    def __len__(self):
        return len(self.images)






                


