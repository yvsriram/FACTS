import torch as ch
import torch.nn as nn
import torchvision.models as tv_models
import os
from collections import OrderedDict

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ResNet(nn.Module):
    def __init__(self, arch, num_classes=10, pretrained=False):
        super(ResNet, self).__init__()
        self.featurizer = getattr(tv_models, arch)(num_classes=1000, pretrained=pretrained)
        self.feature_dim = self.featurizer.fc.in_features
        del self.featurizer.fc
        self.featurizer.fc = Identity()
        self.linear = ch.nn.Linear(self.feature_dim, num_classes)
    def featurize(self, x):
        x = self.featurizer.conv1(x)
        x = self.featurizer.bn1(x)
        x = self.featurizer.relu(x)
        x = self.featurizer.maxpool(x)
        x = self.featurizer.layer1(x)
        x = self.featurizer.layer2(x)
        x = self.featurizer.layer3(x)
        x = self.featurizer.layer4(x)
        return x

    def forward(self, x):
        out = self.featurizer(x)
        out = self.linear(out)
        return out

def construct_model(arch=None, num_classes=10, load_ckpt_path='', pretrained=False):
    if 'resnet' in arch:
        model = ResNet(arch, num_classes=num_classes, pretrained=pretrained)
    else:
        raise NotImplementedError
    if load_ckpt_path != '' and os.path.isfile(load_ckpt_path):
        try:
            model.featurizer.load_state_dict(ch.load(load_ckpt_path))
        except:
            model.load_state_dict(ch.load(load_ckpt_path))
    model = model.cuda()
    return model
