import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
from layers import CNN, HighwayEncoder
from layers import Embedding as wordEmbedding

class MultiheadAttentionLayer(nn.Module):
    
    def __init__(self, hid_dim, num_heads):
        
        super().__init__()
        self.num_heads = num_heads
        self.hid_dim = hid_dim
        
        self.head_dim = self.hid_dim // self.num_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
        
    def forward(self, x, mask):
        # x = [bs, len_x, hid_dim]
        # mask = [bs, len_x]
        
        batch_size = x.shape[0]
        
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)
        # Q = K = V = [bs, len_x, hid_dim]
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        # [bs, len_x, num_heads, head_dim ]  => [bs, num_heads, len_x, head_dim]
        
        K = K.permute(0,1,3,2)
        # [bs, num_heads, head_dim, len_x]
        
        energy = torch.matmul(Q, K) / self.scale
        # (bs, num_heads){[len_x, head_dim] * [head_dim, len_x]} => [bs, num_heads, len_x, len_x]
        
        mask = mask.unsqueeze(1).unsqueeze(2)
        # [bs, 1, 1, len_x]
        
        #print("Mask: ", mask)
        #print("Energy: ", energy)
        
        energy = energy.masked_fill(mask == 0, -1e10)
        
        #print("energy after masking: ", energy)
        
        alpha = torch.softmax(energy, dim=-1)
        #  [bs, num_heads, len_x, len_x]
        
        #print("energy after smax: ", alpha)
        alpha = F.dropout(alpha, p=0.1)
        
        a = torch.matmul(alpha, V)
        # [bs, num_heads, len_x, head_dim]
        
        a = a.permute(0,2,1,3)
        # [bs, len_x, num_heads, hid_dim]
        
        a = a.contiguous().view(batch_size, -1, self.hid_dim)
        # [bs, len_x, hid_dim]
        
        a = self.fc_o(a)
        # [bs, len_x, hid_dim]
        
        #print("Multihead output: ", a.shape)
        return a

class PositionEmbedding(nn.Module):

    #N is a user-defined scalar set by the paper Attention is All you Need
    def __init__(self, hidden_size, device, max_len = 600): 
        super(PositionEmbedding, self).__init__()
        positions = torch.arange(max_len).float().reshape(1, -1).transpose(0, 1)
        i = torch.arange(hidden_size//2)
        denom = ((100000)**(2*i/hidden_size))
        
        Psin = torch.sin(positions / denom)
        Pcos = torch.cos(positions / denom)
        self.positionEncoding = torch.ones(max_len, hidden_size, device=device)

        self.positionEncoding[:, :-1:2] = Psin
        self.positionEncoding[:, 1::2] = Pcos
        

    def forward(self, x):
        # print(self.positionEncoding[:x.shape[1],:].size())
        return x + self.positionEncoding[:x.shape[1],:]


class DepthSepConv(nn.Module):

    def __init__(self, input_size, output_size, kernel_size=3):
        super(DepthSepConv, self).__init__()

        self.depthwise = nn.Conv1d(input_size, input_size, kernel_size, padding=(kernel_size//2), groups=input_size, bias=False)
        self.separate = nn.Conv1d(input_size, output_size, kernel_size=1, bias=True)

    def forward(self, x): #(batch_size, char_embedding_size, max_word_length)
        x_depth = self.depthwise(x.permute(0,2,1))

        ##added relu
        return F.relu(self.separate(x_depth).permute(0, 2, 1))



class EncoderBlock(nn.Module):

    def __init__(self, num_convLayers, kernel_size, hidden_size, num_heads, device, useMask = False):
        super(EncoderBlock, self).__init__()

#         self.positionEncoder = PositionEmbedding(hidden_size)
        
        self.positionEncoder = PositionEmbedding(hidden_size, device)
        
        #create convolutional layers
        self.convLayerNorm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_convLayers)])
        self.convLayer = nn.ModuleList([DepthSepConv(hidden_size, hidden_size, kernel_size=kernel_size) for _ in range(num_convLayers)])
        
        self.selfAttentionNorm = nn.LayerNorm(hidden_size)
        self.selfAttention = MultiheadAttentionLayer(hidden_size, num_heads)
#         self.selfAttention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.forwardLayerNorm = nn.LayerNorm(hidden_size)
        self.forwardLayer = nn.Linear(hidden_size, hidden_size)

        self.K = nn.Linear(hidden_size, hidden_size)
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.useMask = useMask

    def forward(self, x, mask):
        
        x = self.positionEncoder(x)

        # convolution layers
        for convLayer, convLayerNorm in zip(self.convLayer, self.convLayerNorm):
            x_prime = convLayerNorm(x)
            x_prime = convLayer(x_prime)
            x_prime = F.dropout(x_prime, p=0.1)
            x = x + x_prime
       
        # self attention layer
        x_prime = self.selfAttentionNorm(x)
    
        x_prime = self.selfAttention(x_prime, mask)
        x = x + x_prime

        # feed forward layer
        x_prime = self.forwardLayerNorm(x)
        x_prime = F.relu(self.forwardLayer(x_prime))
        x = F.dropout(x + x_prime, p=0.1)

        return x

class ModelEncoder(nn.Module):

    def __init__(self, num_convLayers, kernel_size, hidden_size, num_heads, num_blocks, device, useMask = False):
        super(ModelEncoder, self).__init__()

        self.encoderStack = nn.ModuleList([EncoderBlock(num_convLayers=num_convLayers, kernel_size=kernel_size, hidden_size=hidden_size, device=device, num_heads=num_heads, ) for _ in range(num_blocks)])
        self.attentionResizer = DepthSepConv(4*hidden_size, hidden_size)
        
        
    def forward(self, x, mask):
        M1 = self.attentionResizer(x)

        for block in self.encoderStack:
          M1 = block(M1, mask)
        
        M2 = M1

        for block in self.encoderStack:
            M2 = block(M2, mask)
        
        M3 = M2

        for block in self.encoderStack:
            M3 = block(M3, mask)

        return M1, M2, M3


class QANOutput(nn.Module):

    def __init__(self, hidden_size):
        super(QANOutput, self).__init__()

        self.W1 = nn.Linear(2*hidden_size, 1, bias=False)
        self.W2 = nn.Linear(2*hidden_size, 1, bias=False)
    
    def forward(self, M1, M2, M3, c_mask):

        start = torch.cat([M1, M2], dim=2)
        end = torch.cat([M1, M3], dim=2)

        start = self.W1(start).squeeze()
        end = self.W2(end).squeeze()

        p1 = masked_softmax(start, c_mask, log_softmax=True)
        p2 = masked_softmax(end, c_mask, log_softmax=True)

        return p1, p2

        
        





    














        



