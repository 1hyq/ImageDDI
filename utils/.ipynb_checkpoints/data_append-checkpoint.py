import pandas as pd

def merge_datasets():
    # 读取第一个数据集
    df1 = pd.read_csv("datasets/Deng's_dataset/drug_listxiao.csv")

    # 重命名第一个数据集的列名
    df1 = df1.rename(columns={
        'DB01296': 'drug_id',
        'N[C@H]1C(O)O[C@H](CO)[C@@H](O)[C@@H]1O': 'smiles',
    })

    # 读取第二个数据集
    df2 = pd.read_csv("datasets/Deng's_dataset/drug_smiles.csv")

    # 合并两个数据集
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # 删除合并后的重复数据
    # 确保 'subset' 参数中的列名与您的数据集中的列名相匹配
    combined_df = combined_df.drop_duplicates(subset=['drug_id', 'smiles'])

    # 保存合并后的数据集到新的 CSV 文件
    combined_df.to_csv("datasets/Deng's_dataset/drug_smiles.csv", index=False)
    print("Datasets merged and saved.")

if __name__ == "__main__":
    merge_datasets()
