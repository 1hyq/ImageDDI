
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import load_model, get_support_model_names


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        # define a full connected layer
        self.fc = nn.Linear(input_dim, output_dim)
        self.adjust_dim = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # the output of the full connected layer
        out = F.relu(self.fc(x))

        if x.size() == out.size():
            out = out + x
        else:
            out = out + self.adjust_dim(x)
        return out

class MolDDI(nn.Module):
    def __init__(self, baseModel, num_classes, hidden_dim, vacb_size, max_motif_len):
        super(MolDDI, self).__init__()
        self.image_encoder = nn.Sequential(*list(baseModel.children())[:-1])
        self.max_motif_len = max_motif_len

        # linear map align the visual feature to the position embeddings
        self.image_fc = nn.Linear(512, hidden_dim)
        self.motif_embedding = nn.Embedding(vacb_size + 1, hidden_dim)
        self.motif_seq_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=6
        )
        self.residualblock = ResidualBlock(1024 + 2 * hidden_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self, x1, x2, motif_seq):
        # process the image by forward propagate image encoder
        x1 = self.image_encoder(x1)
        x2 = self.image_encoder(x2)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        # with torch.no_grad():
        img_embed = x1 + x2
        img_embed_transformed = self.image_fc(img_embed)

        motif_embed = self.motif_embedding(motif_seq)
        seq_len = motif_seq.size(1)

        position_embed = img_embed_transformed.unsqueeze(1).expand(-1, seq_len, -1)

        motif_embed = motif_embed + position_embed
        motif_embed = self.motif_seq_encoder(motif_embed)

        embed = torch.mean(motif_embed, dim=1)

        motif_embed_avg = torch.mean(motif_embed, dim=1)

        x_all = torch.cat((x1,x2,embed, motif_embed_avg), dim=1)

        feature = self.residualblock(x_all)
        x = self.fc2(feature)

        return x,feature
