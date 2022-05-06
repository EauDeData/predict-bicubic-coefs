from turtle import forward
import torch


class CoefsModel(torch.nn.Module):
    def __init__(self) -> None:
        super(CoefsModel, self).__init__()

        self.conv = torch.nn.Conv2d(1, 3, 8, stride=1, dilation=2, padding = 7)
        self.conv2 = torch.nn.Conv2d(3, 6, 8, stride=1, dilation=2, padding = 7) # Resulting tensor (16, w, h); the 16 coefs for each pixel
        self.conv3 = torch.nn.Conv2d(6, 12, 8, stride=1, dilation=2, padding = 7)
        self.conv4 = torch.nn.Conv2d(12, 16, 8, stride=1, dilation=2, padding = 7)


        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(0) # We're trying to overfit 1 image and see what happens
        x = self.relu(self.conv(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        return self.conv4(x).view(-1, 4, 4)