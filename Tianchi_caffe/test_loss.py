import torch as t
import torch
import numpy as np
tx=np.random.random_sample(16*1*4*4).reshape([16,1,4,4])
ty=np.random.randint(0,2,16*1*4*4).reshape([16,1,4,4])
import torch as t
#import ipdb;ipdb.set_trace()

def t_dice(x, target):
    # num,loss=0,0
    # for x_,t_ in zip(x,target):
    #     if t_.sum()>40 :
    #         num+=1
    #         loss+=2*(torch.sum(x_*t_))/(torch.sum(x_)+torch.sum(t_))
    # # mask =mask.view(-1,1,1,1,1).expand_as(x)
    batch_size = x.size(0)
    andN = 2 * (torch.sum((x * target).view(batch_size, -1), 1))
    orU1 = torch.sum(x.view(batch_size, -1), 1)
    orU2 = torch.sum(target.view(batch_size, -1), 1)
    dices =  andN / (orU1 + orU2 )
    losses = 1 - dices
    return losses.mean()

def numpy_dice(tx,ty):
    andN=2*np.sum(tx*ty,axis=(1,2,3)).reshape([tx.shape[0],1])
    orU1=np.sum(tx+ty,axis=(1,2,3)).reshape([tx.shape[0],1])
    #orU2=np.sum(ty,axis=1).reshape([tx.shape[0],1])
    dices =  andN / orU1#(orU1 + orU2 )
    losses = 1 - dices
    return losses.mean()
print "t_dice loss:",t_dice(t.from_numpy(tx).float(),t.from_numpy(ty).float())
print "numpy loss:",numpy_dice(tx,ty)


txx=t.autograd.Variable(t.from_numpy(tx).float(),requires_grad=True)
tyy=t.autograd.Variable(t.from_numpy(ty).float(),requires_grad=True)
loss = t_dice(txx,tyy)
loss.backward()
print txx.grad,tyy.grad

def numpy_dice_backpro(tx,ty):
    assert tx.shape[0]==ty.shape[0]
    N=tx.shape[0]
    sums=np.sum(tx+ty,axis=(1,2,3)).reshape([tx.shape[0],1,1,1])
    up=np.sum(tx*ty,axis=(1,2,3)).reshape([tx.shape[0],1,1,1])
    dice=2*up/sums
    x_grad=-1*(2*ty-dice)/sums/N
    y_grad=-1*(2*tx-dice)/sums/N
    return x_grad,y_grad
x_grad,y_grad= numpy_dice_backpro(tx,ty)
from scipy.linalg.misc import norm

print norm(x_grad-txx.grad.data.numpy())/norm(x_grad+txx.grad.data.numpy())
print norm(y_grad-tyy.grad.data.numpy())/norm(y_grad+tyy.grad.data.numpy())
