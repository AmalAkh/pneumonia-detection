import torch
from torch.utils.data import DataLoader
from dataset import PneumoniaDataset
from model import PneumoniaDetectionModel
TRAIN_DATA_DIR = "/Users/amalahmadinurov/Desktop/chest_xray/train"
TEST_DATA_DIR = "/Users/amalahmadinurov/Desktop/chest_xray/train"

#train_dataset =  PneumoniaDataset(TRAIN_DATA_DIR)
test_dataset =  PneumoniaDataset(TEST_DATA_DIR)



#train_dataloader = DataLoader(train_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

model = PneumoniaDetectionModel()
model.fit(test_dataloader)