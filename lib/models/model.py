from collections import OrderedDict
import torch.nn as nn
import torch
import os

class Segmentation_net(nn.Module):
    def __init__(self):
        super(Segmentation_net, self).__init__()

        self.layer0 = self._make_layer(1, 32, conv_nums=2)
        self.layer1 = self._make_layer(32, 64, conv_nums=3)
        self.layer2 = self._make_layer(64, 64, conv_nums=4)
        self.layer3 = self._conv(64, 1024, 15)
        self.layer4 = self._conv(1024, 1, 1)

    def forward(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feature = self.layer3(x)
        out = self.layer4(feature)

        return out, feature

    def _make_layer(self, inplanes, planes, conv_nums, conv_filter=5):

        layers = [self._conv(inplanes, planes, conv_filter)]
        layers += [self._conv(planes, planes, conv_filter)
                   for _ in range(conv_nums - 1)]
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def _conv(self, inplanes, planes, conv_filter):
        pad = int((conv_filter - 1) / 2)
        layers = [nn.Conv2d(inplanes, planes, conv_filter, padding=pad),
                  nn.BatchNorm2d(planes),
                  nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)


class Decision_net(nn.Module):
    def __init__(self, num_classes=1):
        super(Decision_net, self).__init__()

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = self._conv(1025, 8, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = self._conv(8, 16, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = self._conv(16, 32, 5)

        self.avgpool_dec = nn.AdaptiveAvgPool2d([1, 1])
        self.maxpool_dec = nn.AdaptiveMaxPool2d([1, 1])
        self.avgpool_seg = nn.AdaptiveAvgPool2d([1, 1])
        self.maxpool_seg = nn.AdaptiveMaxPool2d([1, 1])

        self.fc_layer = nn.Linear(66, num_classes)

    def forward(self, seg_x, feature):

        # Concat
        x = torch.cat((feature, torch.sigmoid(seg_x)), 1)

        x = self.pool0(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # dec max/avg_pool
        dec_max = self.maxpool_dec(x)
        dec_avg = self.avgpool_dec(x)

        # seg max/avg_pool
        seg_max = self.maxpool_seg(seg_x)
        seg_avg = self.avgpool_seg(seg_x)

        # Full Connection
        cat = torch.cat((dec_max, dec_avg, seg_avg, seg_max), 1).view((-1, 66))
        out = self.fc_layer(cat)
        return out

    def _conv(self, inplanes, planes, conv_filter):
        pad = int((conv_filter - 1) / 2)
        layers = [nn.Conv2d(inplanes, planes, conv_filter, padding=pad),
                  nn.BatchNorm2d(planes),
                  nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)


class SegDecNet(nn.Module):
    def __init__(self, cfg, device, train_type, load_segnet):
        super(SegDecNet, self).__init__()

        self.train_type = train_type

        # SegNet
        self.segnet = Segmentation_net()

        # DecNet
        if self.train_type == 'DecNet':
            self.decnet = Decision_net(num_classes=1)

            if load_segnet:
                print('# --->  Load SegNet')
                # Load SegNet's weight and Freeze it.
                state = torch.load(os.path.join(cfg.CHECKPOINT_PATH, 'SegNet_model_best.pth'), map_location=device)

                # Change name
                new_state_dict = OrderedDict()
                for k, v in state['state_dict'].items():
                    if k[:7] == 'segnet.':
                        name = k[7:]
                        new_state_dict[name] = v

                self.segnet.load_state_dict(new_state_dict, strict=True)
                print('# --->  Load SegNet (Done !!)')

                # Freeze the seg_net
                print('# --->  Freeze SegNet')
                for name, param in self.segnet.named_parameters():
                    param.requires_grad = False
                print('# --->  Freeze SegNet (Done !!)')

                # Check
                for net_name, m in [['segnet', self.segnet], ['decnet', self.decnet]]:
                    for name, param in m.named_parameters():
                        print('{0}.{1} || {2}'.format(net_name, name, param.requires_grad))
                        assert (False if net_name == 'segnet' else True) == param.requires_grad, \
                            'Plz check the freeze layers of SegNet.'

    def forward(self, x):

        dec_out = None

        # SegNet
        seg_out, feature = self.segnet(x)

        # DecNet
        if self.train_type == 'DecNet':
            dec_out = self.decnet(seg_out, feature)

        seg_out = seg_out.squeeze(1)

        return seg_out, dec_out



if __name__ == '__main__':

    from lib.config import cfg

    # input & target
    img = torch.randn(1, 1, 512, 1408)
    tar = torch.rand(1, 1, 64, 176)

    # # SegNet
    # segnet = Segmentation_net()
    # seg_out, feature = segnet(img)
    #
    # # Decnet
    # decnet = Decision_net(num_classes=1)
    # dec_out = decnet(seg_out, feature)

    # SegDecNet
    m = SegDecNet(cfg, device='cpu', train_type='DecNet', load_segnet=True)
    seg_out, dec_out = m(img)

    # Check
    for name, param in m.named_parameters():
        net_name = name.split('.')[0]
        print('{0}.{1} || {2}'.format(net_name, name, param.requires_grad))
        assert (False if net_name == 'segnet' else True) == param.requires_grad, \
            'Plz check the freeze layers of SegNet.'
