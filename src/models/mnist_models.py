import torch.nn as nn

# From https://github.com/ssg-research/WAFFLE/blob/main/src/Network.py
class MNIST_L5(nn.Module):
    def __init__(self, dropout=0.0):
        nn.Module.__init__(self)
        self.dropout = dropout

        self.block = nn.Sequential(
            nn.Conv2d(1, 32, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128 * 5 ** 2, 200)
        self.fc2 = nn.Linear(200, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        out = self.block(x)
        out = out.reshape(-1, 128 * 5 ** 2)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
