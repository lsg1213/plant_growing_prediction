import torch
from torch import nn
from torchvision.models import mobilenet_v2
from efficientnet_pytorch import EfficientNet


class CompareCNN(nn.Module):

    def __init__(self):
        super(CompareCNN, self).__init__()
        self.mobile_net = mobilenet_v2(pretrained=True)
        self.fc_layer = nn.Linear(1000, 1)

    def forward(self, input):
        x = self.mobile_net(input)
        output = self.fc_layer(x)
        return output


def replace_batchnorm2groupnorm(net):
    if len([i for i in net.named_children()]) != 0:
        for name, child in net.named_children():
            if 'bn' in name:
                layer = getattr(net, name)
                chan = layer.num_features
                setattr(net, name, torch.nn.GroupNorm(4, chan))

            replace_batchnorm2groupnorm(child)


class CompareNet(nn.Module):

    def __init__(self, config):
        super(CompareNet, self).__init__()
        model_num = int(config.name.split('_')[0][1:])
        self.config = config
        self.before_net = EfficientNet.from_pretrained(f'efficientnet-b{model_num}')
        self.after_net = EfficientNet.from_pretrained(f'efficientnet-b{model_num}')

        replace_batchnorm2groupnorm(self.before_net)
        replace_batchnorm2groupnorm(self.after_net)
        self.final_fc1 = nn.Linear(2000, 1)
        self.final_fc2 = nn.Linear(2000, 1)
        try:
            getattr(self.config, 'self')
            self.condition = False
        except:
            self.condition = True
        # self.final_fc1 = nn.Linear(256, 1)
        # self.final_fc2 = nn.Linear(256, 1)

        # self.fc11 = nn.Linear(1000, 512)
        # self.fc21 = nn.Linear(1000, 512)
        # self.bc11 = nn.BatchNorm1d(512)
        # self.bc21 = nn.BatchNorm1d(512)
        # self.swish = nn.SiLU()
        # self.fc12 = nn.Linear(512, 256)
        # self.fc22 = nn.Linear(512, 256)
        # self.bc12 = nn.BatchNorm1d(256)
        # self.bc22 = nn.BatchNorm1d(256)

    def forward(self, before_input, after_input):
        before1 = self.before_net(before_input)
        after1 = self.after_net(after_input)
        before2 = self.after_net(before_input)
        after2 = self.before_net(after_input)
        # after = self.before_net(after_input)

        # before = self.fc11(before)
        # before = self.bc11(before)
        # before = self.swish(before)
        # before = self.fc12(before)
        # before = self.bc12(before)
        # before = self.swish(before)

        
        # after = self.fc21(after)  
        # after = self.bc21(after)
        # after = self.swish(after)
        # after = self.fc22(after)
        # after = self.bc22(after)
        # after = self.swish(after)
        if self.condition:
            before = self.final_fc1(torch.cat([before1, before2], -1))
            after = self.final_fc2(torch.cat([after1, after2], -1))
            delta = after - before
            return delta
        else:
            return before1, before2, after1, after2