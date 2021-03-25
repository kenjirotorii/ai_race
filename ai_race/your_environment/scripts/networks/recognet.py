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
        self.bn4 = nn.BatchNorm1d(256)
        self.head = nn.Linear(64, outputs)

    def forward(self, x, m):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn4(self.dense(x)))
        
        x = x.view(-1, 4, 64)
        m = m.view(-1, 1, 4)
        x = torch.matmul(m, x)
        x = x.view(-1, 64)

        x = self.head(x)
        return x


class SeasonNet(nn.Module):
    def __init__(self, h, w, num_classes, num_actions):
        super(SeasonNet, self).__init__()

        self.div = int(h / 2)
        self.num_classes = num_classes
        self.recog = RecogNet(self.div, w, num_classes)
        self.control = ControlNet(self.div, w, num_actions)

    def forward(self, x):
        x1 = x[:, :, :self.div, :]
        x2 = x[:, :, self.div:, :]

        x1 = self.recog(x1)
        mask = F.one_hot(x1.argmax(1), num_classes=self.num_classes).to(torch.float)

        x = self.control(x2, mask)

        return x


if __name__ == "__main__":

    h = 240
    w = 320
    num_classes = 4
    num_actions = 3

    img = torch.rand(4, 3, h, w)

    recog = RecogNet(h / 2, 320, num_classes)

    season = SeasonNet(h, w, num_classes, num_actions)

    season.recog.load_state_dict(recog.state_dict())

    for name, param in season.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name == "recog":
            param.requires_grad = False

    print(season)

    for name, param in season.named_parameters():
        print(name, param.requires_grad)

    # img_u = img[:, :, :120, :]
    # img_b = img[:, :, 120:, :]

    # recog = RecogNet(120, 320, 4)
    # control = ControlNet(120, 320, 3)

    # recog.eval()

    # out1 = recog(img_u)
    # mask = F.one_hot(out1.argmax(1), num_classes=4).to(torch.float)

    # print(out1)
    # print(mask)

    # control.eval()

    # y = control(img_b, mask)

    # print(y)
    