
import torch
import torch.nn as nn

class Model(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        assert "train" in config['datasets'].keys(), "Dataset must contain 'train' key"
        assert "test" in config['datasets'].keys(), "Dataset must contain 'test' key"
        
        # model architecture
        hidden_size = config['model']['hidden_size']
        self.lstm = nn.LSTM(config['model']['in_features'], hidden_size[0], batch_first=True)
        self.fc = nn.Linear(hidden_size[0], hidden_size[1])
        self.module = nn.ModuleList([])
        for i in range(2, len(hidden_size)):
            self.module.append(nn.ReLU())
            self.module.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.module.append(nn.ReLU())
        self.module.append(nn.Linear(hidden_size[-1], config['model']['num_classes']))

        # loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['training']['learning_rate'])

        # datasets and epochs
        self.datasets = config['datasets']
        self.epochs = config['training']['epochs']
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if len(x.shape) != 3:
            x = x.unsqueeze(0)

        intermediate, _ = self.lstm(x)
        intermediate = self.fc(intermediate[:, -1, :])
        for layer in self.module:
            intermediate = layer(intermediate)
        output = nn.functional.softmax(intermediate, dim=1)
        return output
    
    # training loop
    def training_loop(self):
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

    # evaluation loop
    def evaluate(self):
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