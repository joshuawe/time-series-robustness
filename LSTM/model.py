
import torch
import torch.nn as nn

class Model(nn.Module):
    
    def __init__(self, datasets: dict, in_features: int, hidden_size: list, num_classes: int):
        super().__init__()

        assert "train" in datasets.keys(), "Dataset must contain 'train' key"
        assert "test" in datasets.keys(), "Dataset must contain 'test' key"

        self.lstm = nn.LSTM(in_features, hidden_size[0], batch_first=True)
        self.fc = nn.Linear(hidden_size[0], hidden_size[1])
        self.module = nn.ModuleList([])

        for i in range(2, len(hidden_size)):
            self.module.append(nn.ReLU())
            self.module.append(nn.Linear(hidden_size[i-1], hidden_size[i]))

        self.module.append(nn.ReLU())
        self.module.append(nn.Linear(hidden_size[-1], num_classes))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.datasets = datasets
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if len(x.shape) != 3:
            x = x.unsqueeze(0)

        intermediate, _ = self.lstm(x)
        intermediate = self.fc(intermediate[:, -1, :])
        for layer in self.module:
            intermediate = layer(intermediate)
        output = nn.functional.softmax(intermediate, dim=1)
        return output
    
    def train(self, epochs: int):
        for epoch in range(epochs):
            epoch_loss = 0
            for x, y in self.datasets["train"]:
                self.optimizer.zero_grad()
                output = self(x)
                loss = self.criterion(output, y.squeeze()) 
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch: {epoch+1}/{epochs}, loss: {epoch_loss:.4f}")


    def evaluate(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.datasets["test"]:
                output = self(x)
                predicted = torch.argmax(output, dim=1)
                total += output.shape[0]
                correct += (predicted == y.squeeze()).sum().item()
        print(f"Accuracy: {100 * correct / total:.2f}%")