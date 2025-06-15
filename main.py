import torch
from torch.utils.data import DataLoader
from dataset import PneumoniaDataset
from model import PneumoniaDetectionModel
TRAIN_DATA_DIR = "/home/amal/Desktop/chest_xray/train"
TEST_DATA_DIR = "/home/amal/Desktop/chest_xray/train"
VAL_DATA_DIR = "/home/amal/Desktop/chest_xray/train"

train_dataset =  PneumoniaDataset(TRAIN_DATA_DIR)
test_dataset =  PneumoniaDataset(TEST_DATA_DIR)
val_dataset =  PneumoniaDataset(VAL_DATA_DIR)



train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16)
val_dataloader = DataLoader(test_dataset, batch_size=16)

model = PneumoniaDetectionModel()
model.fit(train_dataloader, val_dataloader)
model.evaluate(test_dataloader)
torch.save(model.state_dict(), "./model")
