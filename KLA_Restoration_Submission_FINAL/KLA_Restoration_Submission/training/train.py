# Training reference for the submitted restoration architecture.
# The final evaluator uses run.py and the bundled models/restoration_model.pth.
import os, random, numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import RestorationNet

class PairDataset(Dataset):
    def __init__(self, degraded_dir, gt_dir, patch=64, samples_per_image=8):
        self.d=degraded_dir; self.g=gt_dir; self.names=sorted(os.path.splitext(x)[0] for x in os.listdir(dg:=degraded_dir) if x.endswith('.npy')); self.patch=patch; self.spi=samples_per_image
    def __len__(self): return len(self.names)*self.spi
    def __getitem__(self,i):
        n=self.names[i%len(self.names)]
        lr=np.load(os.path.join(self.d,n+'.npy')).astype('float32')/255
        gt=np.array(Image.open(os.path.join(self.g,n+'.png')).convert('L'),dtype='float32')/255
        p=min(self.patch,lr.shape[0],lr.shape[1]); y=random.randrange(lr.shape[0]-p+1); x=random.randrange(lr.shape[1]-p+1)
        a=lr[y:y+p,x:x+p]; b=gt[2*y:2*y+2*p,2*x:2*x+2*p]
        k=random.randrange(4); a=np.rot90(a,k).copy(); b=np.rot90(b,k).copy()
        if random.random()<.5: a=np.flip(a,0).copy(); b=np.flip(b,0).copy()
        if random.random()<.5: a=np.flip(a,1).copy(); b=np.flip(b,1).copy()
        if random.random()<.35: a += np.random.randn(*a.shape).astype('float32')*random.uniform(.005,.03)
        if random.random()<.35: a += a*np.random.randn(*a.shape).astype('float32')*random.uniform(.01,.05)
        return torch.from_numpy(a[None]),torch.from_numpy(b[None])

def train(degraded_dir,gt_dir,out_path,epochs=60):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=RestorationNet().to(device)
    ds=PairDataset(degraded_dir,gt_dir); dl=DataLoader(ds,batch_size=8,shuffle=True,num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5)
    for e in range(epochs):
        model.train()
        for x,y in dl:
            x,y=x.to(device),y.to(device); p=model(x); loss=nn.functional.l1_loss(p,y)+.25*nn.functional.mse_loss(p,y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(e+1,float(loss))
    torch.save(model.state_dict(),out_path)
