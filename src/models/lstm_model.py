
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Lstm_module(nn.Module):
    def __init__(self, lstm_layers):
        super().__init__()

        self.lstm_module = nn.ModuleList()
        for i in range(1, len(lstm_layers)):
            self.lstm_module.append(nn.LSTM(lstm_layers[i-1], lstm_layers[i], batch_first=True))

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        for lstm in self.lstm_module:
            x, _ = lstm(x)
        return x
    
class FC_module(nn.Module):

    def __init__(self, fc_layers, num_classes):
        super().__init__()

        self.fc_module = nn.Sequential(
            *[nn.Sequential(nn.Linear(fc_layers[i], fc_layers[i+1]), nn.ReLU()) 
                for i in range(len(fc_layers)-1)] 
                + [nn.Linear(fc_layers[-1], num_classes)]
                )

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        x = self.fc_module(x)
        return x



class Model(nn.Module):
    """
    LSTM model for time series classification.

    Args:
        config (dict): A dictionary containing the model configuration.

    Attributes:
        lstm_module (nn.LSTM): The LSTM layer.
        fc_module (nn.Linear): The fully connected layer.
        module (nn.ModuleList): The list of ReLU and Linear layers.
        criterion (nn.CrossEntropyLoss): The loss function.
        optimizer (torch.optim.Adam): The optimizer.
        datasets (dict): A dictionary containing the train and test datasets.
        epochs (int): The number of epochs to train the model.

    Raises:
        AssertionError: If the 'train', 'test' or 'val' key is not present in the dataset.

    """
    def __init__(self, config):
        super().__init__()

        assert "train" in config['datasets'].keys(), "Dataset must contain 'train' key"
        assert "val" in config['datasets'].keys(), "Dataset must contain 'val' key"
        assert "test" in config['datasets'].keys(), "Dataset must contain 'test' key"
        
        # model architecture
        lstm_layers = [config['model']['in_features'], *config['model']['lstm_layers']]
        self.lstm_module = Lstm_module(lstm_layers)

        fc_layers = [config['model']['lstm_layers'][-1], *config['model']['fc_layers']]
        self.fc_module = FC_module(fc_layers, config['model']['num_classes'])

        # loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['training']['learning_rate'])

        # datasets and epochs
        self.datasets = config['datasets']
        self.epochs = config['training']['epochs']
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        if len(x.shape) != 3:
            x = x.unsqueeze(0)

        intermediate = self.lstm_module(x)
        ouput = self.fc_module(intermediate[:, -1, :])
        output = nn.functional.softmax(ouput, dim=1)
        return output.squeeze()
    
    def training_loop(self):
        """
        Training loop for the model.
        """
        for epoch in range(self.epochs):
            epoch_loss = 0
            for x, y in self.datasets["train"]:
                self.optimizer.zero_grad()
                output = self(x)
                loss = self.criterion(output, y.squeeze()) 
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch: {epoch+1}/{self.epochs}, loss: {epoch_loss:.4f}")

    def evaluate(self):
        """
        Evaluation loop for the model.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.datasets["test"]:
                output = self(x)
                predicted = torch.argmax(output, dim=1)
                total += output.shape[0]
                correct += (predicted == y.squeeze()).sum().item()
        accuracy_score = 100 * correct / total
        print(f"Accuracy: {accuracy_score:.2f}%")