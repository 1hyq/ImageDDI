import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rdkit import Chem
import dgl
import random
from rdkit.Chem import Draw
from torch.utils.data import Dataset
from motif_utils import get_substructure_molecule, smiles2graph
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer, mol_to_graph, CanonicalBondFeaturizer, CanonicalAtomFeaturizer, PretrainAtomFeaturizer, PretrainBondFeaturizer
from tqdm import tqdm
from graph_utils import brics_decomp



def compute_combined_distance_matrix(cliques1, edges1, cliques2, edges2):
    num_cliques1 = len(cliques1)
    num_cliques2 = len(cliques2)
    total_cliques = num_cliques1 + num_cliques2
    
    # 初始化距离矩阵，所有距离初始设置为无穷大
    distance_matrix = np.full((total_cliques, total_cliques), 510)
    
     # 识别并处理共同原子团
    common_cliques = identify_common_cliques(cliques1, cliques2)
    for i, j in common_cliques:
        # 对于共同原子团，将它们之间的距离设置为0
        distance_matrix[i, j] = 0
        distance_matrix[j, i] = 0
    
    # 填充第一个分子和第二个分子内部的距离
    for edge in edges1 + [(e[0] + num_cliques1, e[1] + num_cliques1) for e in edges2]:
        i, j = edge
        distance_matrix[i, j] = 1  
        distance_matrix[j, i] = 1
    
   
    
    # 使用Floyd-Warshall算法更新距离矩阵
    for k in range(total_cliques):
        for i in range(total_cliques):
            for j in range(total_cliques):
                # 如果通过k节点的路径更短，则更新距离
                distance_matrix[i, j] = min(distance_matrix[i, j], distance_matrix[i, k] + distance_matrix[k, j])

    return distance_matrix


def pad_cliques(cliques, pad_value=None):
    if pad_value is None:
        pad_value = [0]  # 在函数内部设置默认值
        
    target_sublist_count = 64
    
    # 当前子列表数量
    current_sublist_count = len(cliques)
    
    # 需要额外添加的子列表数量
    if current_sublist_count < target_sublist_count:
        additional_sublist_count = target_sublist_count - current_sublist_count
        additional_cliques = [pad_value for _ in range(additional_sublist_count)]
        cliques.extend(additional_cliques)

    return cliques


def pad_cliques_2(cliques, pad_value=0):
    # 确定所有子列表中的最大长度
    max_length = 32

    # 填充每个子列表至最大长度
    padded_cliques = [sublist + [pad_value] * (max_length - len(sublist)) for sublist in cliques]
    return padded_cliques

    
    
def pad_or_trim_2d(x, max_dim):
    xlen, xdim = x.shape  # 修改为使用shape属性
    if xlen > max_dim:
        x = x[:max_dim, :max_dim]  # 正确裁剪NumPy数组
    elif xlen < max_dim:
        new_x = np.zeros((max_dim, max_dim), dtype=x.dtype)  # 使用np.zeros创建新的NumPy数组
        new_x[:xlen, :xlen] = x
        x = new_x
    return x
    
def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    elif xlen > padlen:
        # 截取x的前padlen行和padlen列
        x = x[:padlen, :padlen]
    return x.unsqueeze(0)
    
def identify_common_cliques(cliques1, cliques2):
    common_cliques = []
    num_cliques1 = len(cliques1)
    num_cliques2 = len(cliques2)
    for index1, clique1 in enumerate(cliques1):
        for index2, clique2 in enumerate(cliques2):
            # 检查两个原子团是否包含完全相同的原子索引集合
            if set(clique1) == set(clique2):
                # 如果相同，将它们添加到common_cliques列表中
                common_cliques.append((index1, index2 + num_cliques1))
    return common_cliques

def pad_tensor_to_fixed_size(tensor, target_size, padding_value=0):
    # 计算需要填充的大小
    padding_needed = target_size - tensor.size(0)
    if padding_needed > 0:
        # 在第一维后面填充
        pad = [0, 0, 0, 0, 0, padding_needed]  # (左, 右, 上, 下, 前, 后)
        tensor = torch.nn.functional.pad(tensor, pad, "constant", padding_value) 
    elif padding_needed < 0:
        # 截断 tensor 到 target_size
        tensor = tensor[:target_size]
    return tensor



class ImageDataset(Dataset):
    def __init__(self, image_folder, txt_file, drug_smiles,motif_seq_file, fold=0, data_type='training', img_transformer=None, normalize=None, ret_index=False, args=None,max_motif_len = 16,max_distance_dim=32):
        '''
        :param image_folder: image path, e.g. ["./drug_image/ImageDDI/DB00001.png", ]
        :param txt_file:drug_id,filename
        :param img_transformer:
        :param normalize:
        :param args:
        '''
        self.args = args
        self.image_folder = image_folder
        self.txt_file = pd.read_csv(txt_file)
        self.drug_smiles=pd.read_csv(drug_smiles)
        path= os.path.join("./datasets/Deng's_dataset/", "{}/ddi_{}1xiao.csv".format(fold, data_type))
        print(path)
        self.normalize = normalize
        self._image_transformer = img_transformer
        self.ddi_data = pd.read_csv(path)
        self.ret_index = ret_index
        self.motif_seq_data = pd.read_csv(motif_seq_file) 
        self.max_motif_len = max_motif_len
        self.max_distance_dim = max_distance_dim
        self.vocab = self.load_vocab(motif_seq_file)
        self.vacb_size = len(self.vocab)


        self.ddi_pairs = {}   
        # 处理 DDI 数据
        self.process_ddi_data()     
    
    
    def load_vocab(self,file_path):
        vocab = {}
        with open(file_path, 'r') as f:
            data = pd.read_csv(f)
            for _, row in data.iterrows():
                motif_id = row['motif_id']  # 获取对应的数字
                order_num = row['order_num']  # 获取化学式子
                vocab[order_num] = motif_id
        return vocab


    def process_ddi_data(self):
        # 创建一个字典来存储 drug ID 对和它们的相互作用标签
        for index, row in self.ddi_data.iterrows():
            self.ddi_pairs[(row['d1'], row['d2'])] = row['type']  
    

    
    
    
    def get_image(self, drug_id):
        try:
        # 检查 drug_id 是否存在于 DataFrame 中
            if not (self.txt_file['drug_id'] == drug_id).any():
                print(f"Drug ID {drug_id} not found in txt_file")
                return None
            image_name = self.txt_file[self.txt_file['drug_id'] == drug_id]['filename'].iloc[0]
            image_path = f"{self.image_folder}/{image_name}"
            img = Image.open(image_path).convert('RGB')
            if self._image_transformer:
                img = self._image_transformer(img)
            return img
        except Exception as e:
            print(f"Error in get_image: {e}")
            return None
    
    
    def __getitem__(self, index): 
        
        keys = list(self.ddi_pairs.keys())
        drug_id1, drug_id2 = keys[index]
        
        label = self.ddi_pairs[(drug_id1, drug_id2)]

        # 加载图片
        img_1 = self.get_image(drug_id1)
        img_2 = self.get_image(drug_id2)
        
        # if img_1 is None or img_2 is None:
        #     print(f"Image not found for drug_id1: {drug_id1} or drug_id2: {drug_id2}")
        # # 返回一个错误标记
        #     return None
        if self.normalize:
            img_1 = self.normalize(img_1)
            img_2 = self.normalize(img_2)
        
        
        # 提取两种药物的 SMILES
        h_smiles = self.drug_smiles[self.drug_smiles['drug_id'] == drug_id1]['smiles'].iloc[0]
        t_smiles = self.drug_smiles[self.drug_smiles['drug_id'] == drug_id2]['smiles'].iloc[0]
        
        
        # mol_graph1 = smiles2graph(h_smiles)
        # mol_graph2 = smiles2graph(t_smiles)
    
        
        h_motifs = get_substructure_molecule(h_smiles, self.vocab)
        t_motifs = get_substructure_molecule(t_smiles, self.vocab)
        motif_seq = list(set(h_motifs).union(set(t_motifs)))
        
        # 使用brics_decomp函数提取原子团和边缘
        h_cliques, h_edges = brics_decomp(h_smiles)
        t_cliques, t_edges = brics_decomp(t_smiles)
        combined_cliques = h_cliques + t_cliques
        
        distance_matrix = compute_combined_distance_matrix(h_cliques, h_edges, t_cliques, t_edges)
        # distance_matrix = distance_matrix.numpy()
        distance_matrix = pad_or_trim_2d(distance_matrix,64)
        distance_matrix = torch.from_numpy(distance_matrix).long()
        # distance_matrix = pad_spatial_pos_unsqueeze(distance_matrix, self.max_distance_dim)      
        # distance_matrix = torch.cat([pad_spatial_pos_unsqueeze(i, 64) for i in distance_matrix])
        # distance_matrix = pad_tensor_to_fixed_size(distance_matrix,32)
        # print(distance_matrix)
        padlen = [0]
        padlen_2 = 0
        combined_cliques = pad_cliques(combined_cliques, padlen)
        combined_cliques = pad_cliques_2(combined_cliques, padlen_2)
        
        # print(combined_cliques)
        # print(len(combined_cliques))
        # if not all(isinstance(item, int) for item in motif_seq):
        #     print("motif_seq contains non-integer values:", motif_seq)
        if len(motif_seq) >= self.max_motif_len:
            motif_seq = motif_seq[:self.max_motif_len]
        else:
            motif_seq.extend([self.vacb_size for _ in range(len(motif_seq), self.max_motif_len)])
            
        motif_seq = torch.tensor(motif_seq,dtype=torch.long) 
        # for i, motif_index in enumerate(motif_seq):
            # print(motif_index)

    
        
        return img_1, img_2,motif_seq,distance_matrix,label
        
        
    def __len__(self):
        return len(self.ddi_pairs)


def load_filenames_and_label(image_folder, txt_file, type_file,task_type="classification"):
    df = pd.read_csv(type_file)
    df_mp=pd.read_csv(txt_file)
    d_1 = df["d1"]
    d_2 = df["d2"]
    labels = df["type"]
    name_mapping = dict(zip(df_mp['drug_id'], df_mp['filename']))
    
    names_1 = [os.path.join(image_folder, name_mapping.get(idx_1) ) for idx_1 in d_1 if name_mapping.get(idx_1) is not None]
    names_2 = [os.path.join(image_folder, name_mapping.get(idx_2) ) for idx_2 in d_2 if name_mapping.get(idx_2) is not None]
    
    return names_1, names_2, labels


def get_datasets(dataset, dataroot, data_type="pretraining"):
   
    image_folder = os.path.join(dataroot, "drug_image/ImageDDI")
    txt_file = os.path.join(dataroot, "drug_image.csv")
    assert os.path.isdir(image_folder), "{} is not a directory.".format(image_folder)
    assert os.path.isfile(txt_file), "{} is not a file.".format(txt_file)
    return image_folder, txt_file


def Smiles2Img(smis, size=224, savePath=None):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png
    '''
    try:
        mol = Chem.MolFromSmiles(smis)
        img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
        if savePath is not None:
            img.save(savePath)
        return img
    except:
        return None

