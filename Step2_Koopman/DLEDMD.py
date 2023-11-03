"""_LSTM class based on torch_
:param param1: a dict content the parameters of MLP architecture and the training parameter

Param={
    'inputsize':M,
    'feature_size':F,
    'outputsize':N,
    'LSTM_hidden':L,
    'Xencodering_layer':[m,m,m],
    'Yencodering_layer':[m,m,m],
    'decodering_layer':[m,m,m],
    'learning_rate':0.001,
    'opt':'Adam',
}

:returns: MLP model
:raises keyError: raises an exception
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.param = params
        
        self.Xencoder = nn.Sequential()
        self.Xencoder.add_module('Xen'+str(0), 
                                 nn.Linear(self.param['X_size'], self.param['Xencodering_layer'][0]))
        self.Xencoder.add_module('Xen_ac'+str(0),nn.ReLU())
        for layer in range(1,len(self.param['Xencodering_layer'])):
            self.Xencoder.add_module('Xen'+str(layer),
                                    nn.Linear(self.param['Xencodering_layer'][layer-1], self.param['Xencodering_layer'][layer]))
            self.Xencoder.add_module('Xen_ac'+str(layer),nn.ReLU())
        self.Xencoder.add_module('Xen_end',nn.Linear(self.param['Xencodering_layer'][-1], self.param['feature_size']))
        
        self.Yencoder = nn.Sequential()
        self.Yencoder.add_module('Yen'+str(0), 
                                 nn.Linear(self.param['Y_size'], self.param['Yencodering_layer'][0]))
        self.Yencoder.add_module('Yen_ac'+str(0),nn.ReLU())
        for layer in range(1,len(self.param['Yencodering_layer'])):
            self.Yencoder.add_module('Yen'+str(layer),
                                    nn.Linear(self.param['Yencodering_layer'][layer-1], self.param['Yencodering_layer'][layer]))
            self.Yencoder.add_module('Yen_ac'+str(layer),nn.ReLU())
        self.Yencoder.add_module('Yen_end',nn.Linear(self.param['Yencodering_layer'][-1], self.param['feature_size']))
        
        
        self.decoder = nn.Sequential()
        self.decoder.add_module('de'+str(0),
                                nn.Linear(self.param['feature_size'], self.param['decodering_layer'][0]))
        self.decoder.add_module('de_ac'+str(0),nn.ReLU(True))
        for layer in range(1,len(self.param['decodering_layer'])):
            self.decoder.add_module('de'+str(layer),
                                    nn.Linear(self.param['decodering_layer'][layer-1], self.param['decodering_layer'][layer]))
            self.decoder.add_module('de_ac'+str(layer),nn.ReLU())
        self.decoder.add_module('de_end',nn.Linear(self.param['decodering_layer'][-1], self.param['Y_size']))
        
        
    def forward(self, Xinputs, Yinputs):
        # feature mapping
        self.faiXu = self.Xencoder(Xinputs)
        self.faiXu = nn.ReLU()(self.faiXu)
        self.faiYu = self.Yencoder(Yinputs)
        self.faiYu = nn.ReLU()(self.faiYu)
        
        self.Gu = torch.matmul(self.faiXu.T, self.faiXu) * (1 / self.param['batch_size'])
        self.Au = torch.matmul(self.faiXu.T, self.faiYu) * (1 / self.param['batch_size'])
        self.Ku = torch.matmul(torch.linalg.inv(self.Gu + 0.001 * torch.eye(self.Gu.shape[0])), self.Au)
        
        feat = torch.matmul(self.faiXu, self.Ku) # 加权求和
        outs = self.decoder(feat)
        outs = nn.ReLU()(outs)
        return outs
    
    