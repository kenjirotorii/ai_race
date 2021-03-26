import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.autoencoder import Autoencoder, Encoder

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
    def __init__(self, h, w, num_z, num_classes, num_actions):
        super(SeasonNet, self).__init__()

        self.div = int(h / 2)
        self.num_classes = num_classes
        self.num_actions = num_actions
        
        self.recog = RecogNet(self.div, w, num_classes)
        self.encoder = Encoder(self.div, w, num_z, False)
        self.head = nn.Linear(num_z, num_actions*num_classes)

    def forward(self, x):
        x1 = x[:, :, :self.div, :]
        x2 = x[:, :, self.div:, :]

        x1 = self.recog(x1)
        mask = F.one_hot(x1.argmax(1), num_classes=self.num_classes).to(torch.float)
        mask = mask.view(-1, 1, self.num_classes)

        x2 = self.encoder(x2)
        x2 = self.head(x2)
        x2 = x2.view(-1, self.num_classes, self.num_actions)

        x = torch.matmul(mask, x2)
        x = x.view(-1, self.num_actions)
        return x


def season_recog_net(img_size, num_z, num_classes, num_actions, ae_model, recog_model):
    
    h, w = img_size
    
    ae = Autoencoder(int(h /2), w, num_z)
    ae.load_state_dict(torch.load(ae_model))
    
    model = SeasonNet(h, w, num_z, num_classes, num_actions)
    
    model.recog.load_state_dict(torch.load(recog_model))
    model.encoder.load_state_dict(ae.encoder.state_dict())

    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name in ["recog", "encoder"]:
            param.requires_grad = False

    return model


if __name__ == "__main__":

    h = 240
    w = 320
    num_classes = 4
    num_actions = 3
    num_z = 256

    img = torch.rand(4, 3, h, w)

    recog = RecogNet(h / 2, w, num_classes)

    ae = Autoencoder(h / 2, w, num_z)

    model = SeasonNet(h, w, num_z, num_classes, num_actions)

    model.recog.load_state_dict(recog.state_dict())
    model.encoder.load_state_dict(ae.encoder.state_dict())

    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name in ["recog", "encoder"]:
            param.requires_grad = False

    print(model)

    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    
    y = model(img)

    print(y) 
