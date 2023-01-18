import torch
from torch import nn
import torch.nn.functional as F


bias_need = False


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(3)

        self.conv_seq1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1, bias=bias_need),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1, bias=bias_need),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_seq2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding=1, bias=bias_need),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1, bias=bias_need),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_seq3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=1, bias=bias_need),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1, bias=bias_need),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1, bias=bias_need),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_seq4 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), padding=1, bias=bias_need),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=bias_need),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=bias_need),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.conv_seq5 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=bias_need),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=bias_need),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1, bias=bias_need),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.dropout = nn.Dropout()
        self.fc_seq1 = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096, bias=bias_need),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.fc_seq2 = nn.Sequential(
            nn.Linear(4096, 4096, bias=bias_need),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.fc_output = nn.Linear(4096, 10)

    def forward(self, x: torch.Tensor):
        n = x.shape[0]
        x = self.bn0(x)
        x = self.conv_seq1(x)
        x = self.conv_seq2(x)
        x = self.conv_seq3(x)
        x = self.conv_seq4(x)
        x = self.conv_seq5(x)
        x = x.reshape((n, -1))
        x = self.dropout(x)
        x = self.fc_seq1(x)
        x = self.fc_seq2(x)
        x = self.fc_output(x)
        return x
