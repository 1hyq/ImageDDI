import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from torch.utils.data import Dataset
from utils.graph_utils import brics_decomp

class DDIDataset(Dataset):
    def __init__(self, image_folder, txt_file, drug_smiles, motif_seq_file, fold =3, data_type='training',
                 img_transformer=None, normalize=None, ret_index=False, args=None, max_motif_len=12, max_distance_dim=32):
        self.args = args
        self.image_folder = image_folder
        self.txt_file = pd.read_csv(txt_file)
        self.drug_smiles = pd.read_csv(drug_smiles)
        path = os.path.join("./datasets/Deng's_dataset/0/", "ddi_{}1xiao.csv".format(data_type))
        self.normalize = normalize
        self._image_transformer = img_transformer
        self.ddi_data = pd.read_csv(path)
        self.ret_index = ret_index
        self.motif_seq_data = pd.read_csv(motif_seq_file)
        self.max_motif_len = max_motif_len
        self.max_distance_dim = max_distance_dim
        self.vocab = self.load_vocab(motif_seq_file)
        self.vacb_size = len(self.vocab)
        self.npz_data = np.load("./data/drug_substructure_dict.npz", allow_pickle=True)[
            'drug_substructure_dict'].item()
        self.ddi_pair = {}
        self.process_ddi_data()

    def load_vocab(self, motif_seq_file):
        vocab = {}
        with open(motif_seq_file, 'r') as f:
            data = pd.read_csv(f)
            for _, row in data.iterrows():
                motif_id = row['motif_id']
                motif = row['motif']
                vocab[motif] = motif_id
        return vocab

    def process_ddi_data(self):
        for index, row in self.ddi_data.iterrows():
            self.ddi_pairs[(row['d1'], row['d2'])] = row['type']

    def get_image(self, drug_id):
        try:
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

    def __getitem__(self, index):
        keys = list(self.ddi_pairs.keys())
        drug_id1, drug_id2 = keys[index]

        label = self.ddi_pairs[(drug_id1, drug_id2)]

        # 加载图片
        img_1 = self.get_image(drug_id1)
        img_2 = self.get_image(drug_id2)

        if img_1 is None or img_2 is None:
            print(f"Image not found for drug_id1: {drug_id1} or drug_id2: {drug_id2}")
            return None

        if self.normalize:
            img_1 = self.normalize(img_1)
            img_2 = self.normalize(img_2)

        # 提取两种药物的 SMILES
        h_smiles = self.drug_smiles[self.drug_smiles['drug_id'] == drug_id1]['smiles'].iloc[0]
        t_smiles = self.drug_smiles[self.drug_smiles['drug_id'] == drug_id2]['smiles'].iloc[0]

        # 从 .npz 文件中加载子结构

        h_motifs = self.npz_data[drug_id1]
        t_motifs = self.npz_data[drug_id2]

        motif_seq = list(set(h_motifs).union(set(t_motifs)))

        if len(motif_seq) >= self.max_motif_len:
            motif_seq = motif_seq[:self.max_motif_len]
        else:
            motif_seq.extend([self.vacb_size for _ in range(len(motif_seq), self.max_motif_len)])

        motif_seq = torch.tensor(motif_seq, dtype=torch.long)

        return img_1, img_2, motif_seq, label

    def __len__(self):
        return len(self.ddi_pairs)

def load_filenames_and_label(image_folder, txt_file, type_file, task_type="classification"):
    df = pd.read_csv(type_file)
    df_mp = pd.read_csv(txt_file)
    d_1 = df["d1"]
    d_2 = df["d2"]
    labels = df["type"]
    name_mapping = dict(zip(df_mp['drug_id'], df_mp['filename']))

    names_1 = [os.path.join(image_folder, name_mapping.get(idx_1)) for idx_1 in d_1 if name_mapping.get(idx_1) is not None]
    names_2 = [os.path.join(image_folder, name_mapping.get(idx_2)) for idx_2 in d_2 if name_mapping.get(idx_2) is not None]

    return names_1, names_2, labels


def get_datasets(dataset, dataroot, data_type="pretraining"):
    image_folder = os.path.join(dataroot, "drug_image/ImageDDI")
    txt_file = os.path.join(dataroot, "drug_image.csv")
    assert os.path.isdir(image_folder), "{} is not a directory.".format(image_folder)
    assert os.path.isfile(txt_file), "{} is not a file.".format(txt_file)
    return image_folder, txt_file
