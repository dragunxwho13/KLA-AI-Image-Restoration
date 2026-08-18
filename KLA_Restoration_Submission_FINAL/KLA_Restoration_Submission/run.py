import argparse, os, glob, numpy as np, torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(c,c,3,padding=1))
    def forward(self,x): return x + 0.1*self.body(x)

class RestorationNet(nn.Module):
    def __init__(self,c=16,blocks=3):
        super().__init__()
        self.head=nn.Conv2d(1,c,3,padding=1)
        self.body=nn.Sequential(*[ResidualBlock(c) for _ in range(blocks)])
        self.fuse=nn.Conv2d(c,c,3,padding=1)
        self.up=nn.Sequential(nn.Conv2d(c,c*4,3,padding=1),nn.PixelShuffle(2),nn.ReLU(inplace=True),nn.Conv2d(c,1,3,padding=1))
    def forward(self,x):
        f=self.head(x); return self.up(f+self.fuse(self.body(f)))

MODEL_PATH=os.path.join(os.path.dirname(__file__),'models','restoration_model.pth')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input_dir'); ap.add_argument('output_dir')
    args=ap.parse_args(); os.makedirs(args.output_dir,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=RestorationNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH,map_location=device,weights_only=True)); model.eval()
    files=sorted(glob.glob(os.path.join(args.input_dir,'*.npy')))
    if not files: raise SystemExit('No .npy files found in input directory.')
    with torch.inference_mode():
        for path in files:
            arr=np.load(path)
            if arr.ndim==3 and arr.shape[-1]==1: arr=arr[...,0]
            if arr.ndim!=2: raise ValueError(f'{os.path.basename(path)}: expected HxW or HxWx1, got {arr.shape}')
            arr=arr.astype(np.float32)
            # Training .npy files are on an approximately 0-255 scale; the released test .npy files are normalized near 0-1.
            scale = 255.0 if np.nanpercentile(np.abs(arr), 99.9) > 2.0 else 1.0
            x=torch.from_numpy(np.ascontiguousarray(arr/scale))[None,None].to(device)
            y=model(x)[0,0].clamp(0,1).cpu().numpy().astype(np.float32)
            y=np.nan_to_num(y,nan=0.0,posinf=1.0,neginf=0.0).clip(0,1).astype(np.float32)
            np.save(os.path.join(args.output_dir,os.path.basename(path)),y)

if __name__=='__main__': main()
