import torch
import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn as nn

class GCNLSTM(nn.Module):
    def __init__(self, input_dim, gcn_dim, seq_length, lstm_dim, output_dim, num_layer):
        super(GCNLSTM, self).__init__()
        #GCN parameter
        self.input_dim = input_dim
        self.gcn_dim = gcn_dim
        #LSTM parameter
        self.lstm_dim = lstm_dim
        self.output_dim = output_dim
        self.num_layer = num_layer
        #the others
        self.seq_length = seq_length
        
        #GCN layer
        self.conv1 = GraphConv(self.input_dim, self.gcn_dim, allow_zero_in_degree=True)
        #LSTM layer
        self.lstm = nn.LSTM(self.gcn_dim, self.lstm_dim,num_layer,batch_first = True)
        self.fc = nn.Linear(self.lstm_dim, self.output_dim)
        
    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.ndata['x']
        # Perform graph convolution and activation function.
        h = torch.relu(self.conv1(g, h))
        #h = torch.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')

        dat = build_dataset(hg, self.seq_length)
        dat = torch.FloatTensor(dat)
        
        
        h_0 = torch.zeros(self.num_layer, dat.size(0), self.lstm_dim).requires_grad_()
        c_0 = torch.zeros(self.num_layer, dat.size(0), self.lstm_dim).requires_grad_()
        
        output, (h_n, c_n) = self.lstm(dat, (h_0, c_0))
        h_n = h_n.view(-1, self.lstm_dim)
        pred = self.fc(h_n)
        
        return pred

