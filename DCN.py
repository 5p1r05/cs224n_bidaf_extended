import torch
import torch.nn as nn
from util import masked_softmax

# Dynamic Coattention Network
class DCN(nn.Module):


    def __init__(self, hidden_size):
        super(DCN, self).__init__()
        self.q_weight = nn.Linear(2*hidden_size, 2*hidden_size, bias=True)
        self.tanh = nn.Tanh()

        nn.init.xavier_uniform_(self.q_weight)
    
    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        q_tanh = self.tanh(self.q_weight(q))

        #compute affinity matrix
        
        L = torch.bmm(q_tanh, torch.permute(c, (0, 2, 1))) #(batch_size, q_len, c_len)

        Aq = masked_softmax(L, q_mask, dim=2)   # (batch_size, c_len, q_len)
        Ac = masked_softmax(L, c_mask, dim=1)       # (batch_size, c_len, q_len)

        C2Q = torch.bmm(torch.permute(Aq, (0, 2, 1)), q)  #(batch_size, c_len, hidden_size)

        Q2C = torch.bmm(Ac, c) #(batch_size, q_len, hidden_size)

        Q2C_attended = torch.bmm(torch.permute(Aq, (0, 2, 1)), Q2C) #(batch_size, c_len, hidden_size)

        coattention_output = torch.cat((Q2C_attended, C2Q), 2)

        return coattention_output





