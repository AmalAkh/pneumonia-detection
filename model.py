import torch
from torch.nn import Linear, Conv2d, ReLU, Sigmoid, AvgPool2d, Flatten
from device import device_name
class PneumoniaDetectionModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.BCELoss()
        
        self.conv_model = torch.nn.Sequential(
            Conv2d(1,32, 3, padding="same"),
            ReLU(),
            Conv2d(32,32, 3, padding="same"),
            ReLU(),
            AvgPool2d((2,2)),
            Conv2d(32,64, 3,padding="same"),
            ReLU(),
            Conv2d(64,64, 3,padding="same"),
            ReLU(),
            AvgPool2d((2,2))
        )
        
        self.classifier = torch.nn.Sequential(
            Flatten(),
            Linear(32*32*64,256),
            ReLU(),
            Linear(256,128),
            ReLU(),
            Linear(128, 1),
            Sigmoid()
        )
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.00001)
       
        self.to(device_name)
    def forward(self, images):
        features = self.conv_model(images)
        
        output = self.classifier(features)
        return output
    def evaluate(self, data_loader):
        self.eval()
        right_predictions = 0
        overall_loss = 0
        with torch.no_grad():
            for batch in iter(data_loader):
                
                x = batch[0].to(device_name)
                y = batch[1].to(device_name).float()

                output = self.forward(x)
                output = output.view(output.size(0), 1)
                
                loss = self.loss_fn(output, y)
                
                overall_loss += loss.item()
                
                right_predictions += (output.round() == y).sum().item()

        print(f"Val Loss:{overall_loss/len(data_loader):.4f} Val Accracy:{right_predictions/len(data_loader.dataset):.4f}")
        
    def fit(self, train_loader, val_loader, epochs=10):
        self.train(True)
        for i in range(0,epochs):
            epoch_loss = 0
            right_predictions = 0
            for batch in iter(train_loader):
                
                x = batch[0].to(device_name)
                y = batch[1].to(device_name).float()

                output = self.forward(x)
                output = output.view(output.size(0), 1)
                
                loss = self.loss_fn(output, y)
                
                epoch_loss += loss.item()
                
                right_predictions += (output.round() == y).sum().item()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
            self.train(False)
            val_epoch_loss = 0
            val_right_predictions = 0
            with torch.no_grad():
                for batch in iter(val_loader):
                    
                    x = batch[0].to(device_name)
                    y = batch[1].to(device_name).float()

                    output = self.forward(x)
                    output = output.view(output.size(0), 1)
                    
                    val_loss = self.loss_fn(output, y)
                    
                    val_epoch_loss += val_loss.item()
                    
                    val_right_predictions += (output.round() == y).sum().item()
                
            self.train(True)
            print(f"Epoch {i+1} Loss:{epoch_loss/len(train_loader):.4f} Accuracy:{right_predictions/len(train_loader.dataset):.4f}, Val Loss:{val_epoch_loss/len(train_loader):.4f} Val Accracy:{val_right_predictions/len(val_loader.dataset):.4f}")
            torch.cuda.empty_cache()
    
        



            