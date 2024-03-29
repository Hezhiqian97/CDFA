import torch
from torch import nn, Tensor, LongTensor
class CDFA(nn.Module):
    def __init__(self, in_channels, kernel_size=3,):
        super(CDFA, self).__init__()

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1,)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1,)
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.cavg_pool = nn.AdaptiveAvgPool2d(1)
        self.cmax_pool = nn.AdaptiveMaxPool2d(1)
        self.c1=nn.Conv2d(in_channels*2, in_channels//2, kernel_size=1,bias=False)
        self.relu1=nn.ReLU()
        self.c2 = nn.Conv2d(in_channels//2,in_channels,  kernel_size=1,bias=False)
        self.c3 = nn.Conv2d(in_channels , in_channels, kernel_size=3, padding=1,bias=False)
        self.c4 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.c5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.c6 = nn.Conv2d(in_channels, in_channels, kernel_size=1)


    def forward(self, x):
        batch_size, channels, height, width = x.size()
        q=self.query_conv(x)
        k=self.key_conv(x)
        v=self.value_conv(x)
        avg_out = torch.mean(q, dim=1, keepdim=True)
        max_out, _ = torch.max(q, dim=1, keepdim=True)
        s = torch.cat([avg_out, max_out], dim=1)
        s = self.conv1(s)
        #print(s.size())
        c_avg_out=self.cavg_pool(k)
        c_max_out=self.cmax_pool(k)
        #print(c_avg_out.size(),c_max_out.size())
        c = torch.cat([c_avg_out, c_max_out], dim=1)
        c = self.c2(self.relu1(self.c1(c)))
        #c_avg_out = self.c2(self.relu1(self.c1(self.cavg_pool(k))))
        #c_max_out = self.c2(self.relu1(self.c1(self.cmax_pool(k))))

        #print(c.size())
        s = s.view(batch_size, -1, 1 * 1).permute(0, 2, 1)
        c=c.view(batch_size, -1, 1 * 1)
        #print(s.size(), c.size())
        out=torch.bmm(c,s)
        #print(out.size())
        out = out.view(batch_size, channels, height, width)
        out = self.c4(torch.cat([v, out], dim=1))
        out=self.relu1(out)
        #print(out.size())
        out = self.c5(out)
        out = self.relu1(out)
        out = self.c6(out)
        out=self.sig(out)
        #print(out.size())
        return x*out