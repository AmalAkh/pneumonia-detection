import torch
from torch.nn import Linear, Conv2d, ReLU, Sigmoid
from device import device_name
class PneumoniaDetectionModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.BCELoss()
        
        self.conv_model = torch.nn.Sequential(
            Conv2d(1,32, 3, padding="same"),
            ReLU(),
            Conv2d(32,32, 3,padding="same"),
            ReLU(),
        )
        
        self.classifier = torch.nn.Sequential(
            Linear(200*200*32,256),
            ReLU(),
            Linear(256, 1),
            Sigmoid()
        )
        self.optimizer = torch.optim.Adam(self.parameters())
    def forward(self, images):
        features = self.conv_model(images)
        flattened_features = features.view(features.size(0), -1)
        output = self.classifier(flattened_features)
        return output
    def fit(self, train_loader, epochs=10):
        
        for i in range(0,epochs):
            epoch_loss = 0
            for batch in iter(train_loader):
                
                x = batch[0].to(device_name)
                y = batch[1].to(device_name)

                output = self.forward(x)
                loss = self.loss_fn(output, y)
                
                epoch_loss += loss.item()
                
                print(epoch_loss)
                self.optimizer.zero_grad()

                loss.backward()
            print(epoch_loss)



            