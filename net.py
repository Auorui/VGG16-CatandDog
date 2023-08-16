import torch
import torchvision
import torch.nn as nn
import torchsummary

from torch.hub import load_state_dict_from_url
# model = torchvision.models.vgg16()

model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
}
cfgs = {
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class VGG(nn.Module):
    def __init__(self, features, num_classes = 1000, init_weights= True, dropout = 0.5):
        super(VGG,self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers_with_vgg(cfg, batch_norm = False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3,3), padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg(mode='vgg16',pretrained=False, progress=True, num_classes=2):
    model = VGG(make_layers_with_vgg(cfgs[mode]))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[mode], model_dir='./model', progress=progress)#预训练模型地址
        model.load_state_dict(state_dict)
    if num_classes != 1000:
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
    return model


if __name__=='__main__':
    in_data = torch.ones(2, 3, 224, 224)
    net = vgg(mode='vgg16', pretrained=False, progress=True, num_classes=2)  # 使用默认的 VGG-16 架构
    # net = vgg(mode='vgg19', pretrained=False, progress=True, num_classes=2)  # 使用 VGG-19 架构
    out = net(in_data)
    print(out)
    torchsummary.summary(net, input_size=(3, 224, 224))
