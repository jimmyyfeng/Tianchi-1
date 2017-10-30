from itertools import chain
import torch
import numpy as np
class Visualizer():
    def __init__(self, env='cysb'):
        import visdom
        self.vis = visdom.Visdom()
        self.index={}
        self.env=env
    def plot(self,name,y):
        x = self.index.get(name,0)
        self.vis.line(Y=np.array([y]),X=np.array([x]),\
                    win=unicode(name),\
                    opts=dict(title=name),\
                    env=self.env,
                    update=None if x==0 else 'append'
        )
        self.index[name] = x + 1

    def img_del(self,input,output,nodules):
        
        input=input.squeeze()
        output=output.squeeze()
        a1,a2,a3=input.size()
        number, index = output.data.sum(-1).sum(-1).view(-1).max(0)
        self.vis.image(input[index[0]].unsqueeze(0),expand(3,a2,a3),win=u'input',\
            opts=dict(title='input of maxoutput'))
        self.vis.image(output[index[0]].unsqueeze(0),expand(3,a2,a2),win=u'max_output',\
            opts=dict(title='max output')
        )
        self.vis.image(input)

        output_max_index=output.data.squeeze().sum(1).view(-1).max(0)
        #[index[0]].unsqueeze(0).expand(0,3)
        self.vis.image(output[output_max_index].unsqueeze(0).expand(0,3))
        self.vis.image()
        self.vis.image(output[index[0]].unsqueeze(0).expand(0,3))
    def img(self, name, img):
        a,b=img.size()
        self.vis.image(img.unsqueeze(0).expand(3,a,b), 
                win=unicode(name),
                env=self.env,
                opts=dict(title=name)
        )
    def __getattr__(self,name):
        return getattr(self.vis, name)
        



