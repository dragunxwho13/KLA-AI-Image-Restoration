import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self,c=16): super().__init__(); self.body=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(c,c,3,padding=1))
    def forward(self,x): return x+.1*self.body(x)
class RestorationNet(nn.Module):
    def __init__(self,c=16,blocks=3):
        super().__init__(); self.head=nn.Conv2d(1,c,3,padding=1); self.body=nn.Sequential(*[ResidualBlock(c) for _ in range(blocks)]); self.fuse=nn.Conv2d(c,c,3,padding=1); self.up=nn.Sequential(nn.Conv2d(c,c*4,3,padding=1),nn.PixelShuffle(2),nn.ReLU(inplace=True),nn.Conv2d(c,1,3,padding=1))
    def forward(self,x):
        f=self.head(x); return self.up(f+self.fuse(self.body(f)))
