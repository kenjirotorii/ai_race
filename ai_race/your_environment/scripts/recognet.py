import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride  + 1


class RecogNet(nn.Module):
    def __init__(self, h, w, outputs):
        super(RecogNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        linear_input_size = convw * convh * 32

        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class ControlNet(nn.Module):
    def __init__(self, h, w, outputs):
        super(ControlNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        linear_input_size = convw * convh * 32

        self.dense = nn.Linear(linear_input_size, 256)
        self.head = nn.Linear(64, outputs)

    def forward(self, x, m):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        
        x = x.view(-1, 4, 64)
        m = m.view(-1, 1, 4)
        x = torch.matmul(m, x)
        x = x.view(-1, 64)

        x = self.head(x)
        return x


if __name__ == "__main__":

    h = 240
    w = 320

    img = torch.rand(4, 3, h, w)
    img_u = img[:, :, :120, :]
    img_b = img[:, :, 120:, :]

    recog = RecogNet(120, 320, 4)
    control = ControlNet(120, 320, 3)

    recog.eval()

    out1 = recog(img_u)
    mask = F.one_hot(out1.argmax(1), num_classes=4).to(torch.float)

    print(out1)
    print(mask)

    control.eval()

    y = control(img_b, mask)

    print(y)
    