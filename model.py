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


class CompareNet(nn.Module):

    def __init__(self, config):
        super(CompareNet, self).__init__()
        model_num = int(config.name.split('_')[0][1:])
        self.before_net = EfficientNet.from_pretrained(f'efficientnet-b{model_num}')
        self.after_net = EfficientNet.from_pretrained(f'efficientnet-b{model_num}')
        
        self.final_fc1 = nn.Linear(2000, 1)
        self.final_fc2 = nn.Linear(2000, 1)
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

        before = self.final_fc1(torch.cat([before1, before2], -1))
        after = self.final_fc2(torch.cat([after1, after2], -1))
        delta = after - before
        return delta