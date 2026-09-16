import sys
sys.path.insert(0, '/home/data/xyz/ABot-Recon')
import numpy as np
from pathlib import Path
from scripts.align_method2_lightglue_umeyama import umeyama_svd
r=Path('/home/data/xyz/ABot-Recon/outputs')
paths=[]
for sid in ['测试_1','测试_2']:
    p=np.load(r/sid/'camera_poses.npy')[:,:3,3]
    d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    u=np.linspace(0,d[-1],40)
    paths.append(np.stack([np.interp(u,d,p[:,k]) for k in range(3)],1))
for rev in [False,True]:
    a,b=paths[0],paths[1][::-1] if rev else paths[1]
    s,R,t=umeyama_svd(a,b,True)
    e=np.linalg.norm(s*(a@R.T)+t-b,axis=1)
    print('reverse',rev,'scale',s,'rmse',np.sqrt(np.mean(e*e)),'mapped endpoints',s*(a[[0,-1]]@R.T)+t,'target',b[[0,-1]])
