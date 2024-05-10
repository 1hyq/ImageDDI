import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn import GCN
import torch.nn as nn
import dgl
from model.cnn_model_utils import load_model, get_support_model_names

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResidualLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualLayer, self).__init__()
        # 定义一个全连接层
        self.fc = nn.Linear(input_dim, output_dim)
        self.adjust_dim = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # 全连接层的输出
        out = F.relu(self.fc(x))
        
        # 残差连接
        # 只有当输入和输出维度相同时才能直接相加
        if x.size() == out.size():
            out = out + x
        else:
            # 如果维度不同，可以通过一个额外的线性变换调整维度
            # 这里需要定义一个额外的线性层用于维度匹配（可以在 __init__ 中定义）
            # 例如: self.adjust_dim = nn.Linear(input_dim, output_dim)
            out = out + self.adjust_dim(x)
        return out
    
    
class MolDDI(nn.Module):
    def __init__(self, baseModel, num_classes, hidden_dim, vacb_size, num_cliques_types,max_motif_len,max_distance_dim):
        super(MolDDI, self).__init__()
        self.image_encoder=nn.Sequential(*list(baseModel.children())[:-1])
        self.max_motif_len=max_motif_len
        self.max_distance_dim = max_distance_dim
        
        self.cliques_embedding = nn.Embedding(num_cliques_types + 1, hidden_dim)
        self.motif_embedding = nn.Embedding(vacb_size+1, hidden_dim)
        self.distance_embedding = nn.Embedding(max_distance_dim * max_distance_dim, hidden_dim) 
        self.cliques_seq_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4), 
            num_layers=4
        )
        self.motif_seq_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4), 
            num_layers=4
        )
        
        self.residual1 = ResidualLayer(1024 + 2*hidden_dim, 512)
        self.fc1 = nn.Linear(1024 + 2*hidden_dim, 512)
        self.fc2 = nn.Linear(512, num_classes) 
        
    
   
    def forward(self, x1, x2, motif_seq,distance_matrix):
        # 处理图像
        x1 = self.image_encoder(x1)
        x2 = self.image_encoder(x2)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
      
        motif_embed = self.motif_embedding(motif_seq)
        motif_embed = self.motif_seq_encoder(motif_embed)
        distance_embed = self.distance_embedding(distance_matrix)
        distance_embed = distance_embed.mean(dim=1)  # 假设简单平均处理
        motif_embed = motif_embed.mean(dim=1)  # 变为 [128, 512]
        distance_embed = distance_embed.mean(dim=1)  # 变为 [128, 512]

        # distance_embed = distance_embed.squeeze(1)
        # print("x1 shape:", x1.shape)
        # print("x2 shape:", x2.shape)
        # print("motif_embed shape:", motif_embed.shape)
        # print("distance_embed shape:", distance_embed.shape)


        x_all = torch.cat((x1, x2, motif_embed, distance_embed), dim=1)
        

        x = self.residual1(x_all)
        x = self.fc2(x)
        

        return x

    
    
# initializing weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
        
def pad_or_trim_2d(x, max_dim):
    xlen, xdim = x.size()
    if xlen > max_dim:
        x = x[:max_dim, :max_dim]  # 如果矩阵过大，进行剪裁
    elif xlen < max_dim:
        new_x = x.new_zeros([max_dim, max_dim], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x


        
'''
        # 处理分子图
        h = mol_graph1.ndata['feat'].float()
        mol_graph1.ndata['feat'] = self.mol_graph_encoder(mol_graph1, h) + self.feature_transformer(h)
        graph_embed1 = dgl.readout_nodes(mol_graph1, 'feat')

        t = mol_graph2.ndata['feat'].float()
        mol_graph2.ndata['feat'] = self.mol_graph_encoder(mol_graph2, t) + self.feature_transformer(t)
        graph_embed2 = dgl.readout_nodes(mol_graph2, 'feat')
'''