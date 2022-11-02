import torch
import torch.nn as nn

def conv3x3(in_channel, out_channel, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, se=False, cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEblock(planes) if se else nn.Identity()
        self.cbam = CBAM(planes) if cbam else nn.Identity()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se(out)
        out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, se=False, cbam=False):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEblock(planes * Bottleneck.expansion) if se else nn.Identity()
        self.cbam = CBAM(planes * Bottleneck.expansion) if cbam else nn.Identity()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se(out)
        out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1, bias=True, norm_layer=nn.BatchNorm2d, act=True):
        super(ConvNormAct,self).__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias),
            norm_layer(out_ch) if norm_layer != nn.Identity() else nn.Identity(),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )



class SEblock(nn.Sequential):
    def __init__(self, channel, r=16):
        super(SEblock, self).__init__(
            # squeeze
            nn.AdaptiveAvgPool2d(1), 

            # excitation
            ConvNormAct(channel, channel//r, 1),
            nn.Conv2d(channel//r, channel, 1, bias=True),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        out = super(SEblock, self).forward(x)
        return x + out



class CBAM(nn.Module):
    def __init__(self, channel, r=16):
        super(CBAM, self).__init__()
        self.avg_channel = nn.AdaptiveAvgPool2d(1)
        self.max_channel = nn.AdaptiveMaxPool2d(1)
        self.shared_excitation = nn.Sequential(
            ConvNormAct(channel, channel//r, 1, bias=False, norm_layer=nn.Identity),
            nn.Conv2d(channel//r, channel, 1, bias=False)
        )
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=7//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        ## channel attention ##
        out1 = self.avg_channel(input)
        out1 = self.shared_excitation(out1)
        out2 = self.max_channel(input)
        out2 = self.shared_excitation(out2)
        channel_attention = nn.Sigmoid()(out1+out2) # (batch, channel, 1, 1)
        input = input * channel_attention

        ## spatial attention ##
        batch, size,_,_ = input.shape
        avg_spatial = input.mean(dim=1).reshape(batch, 1, size, -1) # (batch, 1, H, W)
        max_spatial = input.max(dim=1)[0].reshape(batch, 1, size, -1) # (batch, 1, H, W)
        spatial_attention = torch.cat([avg_spatial, max_spatial], 1)
        spatial_attention = self.conv_spatial(spatial_attention)
        input = input * spatial_attention

        return input



import torch
import torch.nn as nn
from timm import create_model


class WongisMIL(nn.Module):

    def __init__(
        self, 
        model_name, 
        num_instances= 4, 
        num_classes=1, 
        pretrained=True):
        super().__init__()

        self.num_instances = num_instances
        self.encoder = create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes)

        enc_type = self.encoder.__class__.__name__

        feature_dim = self.encoder.get_classifier().in_features
        self.tabular_feature_extractor = TabularFeatureExtractor()

        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), 
            Flatten(),
            nn.Linear(feature_dim, 512),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Linear(in_features=512, out_features=1),
            # nn.Sigmoid(),
        )

    def forward(self, image, tabular):
        # x: bs x N x C x W x W
        bs, _, ch, w, h = image.shape
        # 2, 3, 3, 256, 256
        x = image.view(bs*self.num_instances, ch, w, h) # x: N bs x C x W x W
        x = self.encoder.forward_features(x) # x: N bs x C' x W' x W'

        # Concat and pool
        bs2, ch2, w2, h2 = x.shape
        x = x.view(-1, self.num_instances, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
            .contiguous().view(bs, ch2, self.num_instances*w2, h2) # x: bs x C' x N W'' x W''
        x = self.head(x)
        tabular_feature = self.tabular_feature_extractor(tabular)
        feature = torch.cat([x, tabular_feature], dim=-1)
        output = self.classifier(feature)

        return output

class Flatten(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x): 
        input_shape = x.shape
        output_shape = [input_shape[i] for i in range(self.dim)] + [-1]
        return x.view(*output_shape)

class TabularFeatureExtractor(nn.Module):
    def __init__(self):
        super(TabularFeatureExtractor, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_features=23, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=512)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        return x