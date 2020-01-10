import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.autograd import Function


# = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class MyNet(nn.Module):

    def __init__(self, num_classes=7):
        super(MyNet, self).__init__()
        self.gf = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.gy = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.gd = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )

    def forward(self, x, alpha=None):
        features = self.gf(x)
        # Flatten the features:
        features = features.view(features.size(0), -1)
        # If we pass alpha, we can assume we are training the discriminator
        if alpha is not None:
            # gradient reversal layer (backward gradients will be reversed)
            reverse_feature = ReverseLayerF.apply(features, alpha)

            discriminator_output = self.gd(reverse_feature)
            return discriminator_output
        # If we don't pass alpha, we assume we are training with supervision
        else:

            class_outputs = self.gy(features)
            return class_outputs


def mynet(progress=True, **kwargs):

    model = MyNet(**kwargs)

    state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress,)
    model.load_state_dict(state_dict, strict=False)
    model.gd[1].weight.data = model.gy[1].weight.data
    model.gd[1].bias.data = model.gy[1].bias.data
    model.gd[4].weight.data = model.gy[4].weight.data
    model.gd[4].bias.data = model.gy[4].bias.data
    model.gd[6].weight.data = model.gy[6].weight.data
    model.gd[6].bias.data = model.gy[6].bias.data

    return model
