import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size=5, stride=2):
    return (size - (kernel_size - 1) - 1) // stride  + 1


class Encoder(nn.Module):
    def __init__(self, h, w, outputs, variational=False):
        super(Encoder, self).__init__()

        self.variational = variational
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        linear_input_size = convw * convh * 32

        if self.variational:
            self.mu_head = nn.Linear(linear_input_size, outputs)
            self.lnvar_head = nn.Linear(linear_input_size, outputs)
        else:
            self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        if self.variational:
            mu = self.mu_head(x)
            lnvar = self.lnvar_head(x)
            return mu, lnvar
        else:
            x = self.head(x)
            return x


class Decoder(nn.Module):
    def __init__(self, inputs, h, w):
        super(Decoder, self).__init__()

        self.convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 4), 4), 4)
        self.convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 4), 4), 4)
        linear_input_size = self.convw * self.convh * 32
        
        self.dense = nn.Linear(inputs, linear_input_size)
        self.conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                        kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=16,
                                        kernel_size=4, stride=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(16)             
        self.conv3 = nn.ConvTranspose2d(in_channels=16, out_channels=3,
                                        kernel_size=4, stride=2)

        self.sigmoid = nn.Sigmoid()
                                
    def forward(self, x):
        x = self.dense(x)
        x = x.view(x.size(0), 32, self.convh, self.convw)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.sigmoid(self.conv3(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, h, w, outputs):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(h, w, outputs, False)
        self.decoder = Decoder(outputs, h, w)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self, h, w, outputs, train_mode=True):
        super(VAE, self).__init__()
        
        self.train_mode = train_mode
        self.encoder = Encoder(h, w, outputs, True)
        self.decoder = Decoder(outputs, h, w)
    
    def reparameterize(self, mu, lnvar):
        std = torch.exp(0.5*lnvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, lnvar = self.encoder(x)
        x = self.reparameterize(mu, lnvar)
        x = self.decoder(x)
        if self.train_mode:
            return x, mu, lnvar
        else:
            return x


class ControlHead(nn.Module):
    def __init__(self, inputs, outputs):
        super(ControlHead, self).__init__()

        self.dense = nn.Linear(inputs, 32)
        self.bn = nn.BatchNorm1d(32)
        self.head = nn.Linear(32, outputs)

    def forward(self, x):
        x = F.relu(self.bn(self.dense(x)))
        x = self.head(x)
        return x


class ControlNet(nn.Module):
    def __init__(self, h, w, z, outputs, variational=True):
        super(ControlNet, self).__init__()

        self.encoder = Encoder(h, w, z, variational)
        self.decoder = ControlHead(z, outputs)

    def reparameterize(self, mu, lnvar):
        std = torch.exp(0.5*lnvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, lnvar = self.encoder(x)
        x = self.reparameterize(mu, lnvar)
        x = self.decoder(x)
        return x


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, y_pred, y_true, mu, lnvar):
        bce = F.binary_cross_entropy(y_pred, y_true, reduction="sum")
        kl = -0.5 * torch.sum(1 + lnvar - mu.pow(2) - lnvar.exp())
        return bce + kl


if __name__ == "__main__":

    h = 120
    w = 320
    img = torch.rand(2, 3, h, w)
    z = 64
    variational = True

    if variational:
        mod = VAE(h, w, z)
    else:
        mod = Autoencoder(h, w, z)

    def example(model, x):
        print(model)
        model.eval()
        if variational:
            y, _, _ = model(x)
        else:
            y = model(x)
        print(y.size())
    
    def transfer(model, x):
        if variational:
            model_new = VAE(h, w, z)
        else:
            model_new = Autoencoder(h, w, z)

        model_new.load_state_dict(model.state_dict())

        for name, param in model_new.named_parameters():
            layer_name = name.split('.')[0]
            if layer_name == "encoder":
                param.requires_grad = False

        model_new.decoder = ControlHead(z, 3)
        print(model_new)

        for name, param in model_new.named_parameters():
            print(name, param.requires_grad)

        model_new.eval()
        if variational:
            y, _, _ = model_new(x)
        else:
            y = model_new(x)

        print(y.size())
    
    # example(mod, img)
    transfer(mod, img)
