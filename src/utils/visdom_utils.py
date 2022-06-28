import torch
import math
import numpy as np
from visdom import Visdom

def new_pane(env, title):
    pane = env.line(
        X=torch.FloatTensor([0]),
        Y=torch.FloatTensor([0]),
        opts=dict(title=title))
    return pane

def append2pane(x, y, env, pane):
    env.line(
        X=torch.FloatTensor([x]),
        Y=torch.FloatTensor([y]),
        win=pane,#win参数确认使用哪一个pane
        update='append') #我们做的动作是追加，除了追加意外还有其他方式，这里我们不做介绍了

