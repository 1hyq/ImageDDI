import json
import csv
import os


class DDIProcessor:
    def __init__(self, task_dir):
        self.task_dir = task_dir
        self.id2entity = {}
        self.id2relation = {}

    def load_ent_id(self):
        id2entity = dict()
        id2relation = dict()

        # drug_set = json.load(open(os.path.join(self.task_dir, 'node2id.json'), 'r'))
        # entity_set = json.load(open(os.path.join(self.task_dir, 'entity_drug.json'), 'r'))
        # relation_set = json.load(open(os.path.join(self.task_dir, 'relation2id.json'), 'r'))

        node2id_path = os.path.join('node2id.json')
        entity_drug_path = os.path.join('entity_drug.json')
        relation2id_path = os.path.join('relation2id.json')

        print("Current working directory:", os.getcwd())
        print("Absolute task directory:", self.task_dir)
        print("node2id.json path:", node2id_path)
        print("entity_drug.json path:", entity_drug_path)
        print("relation2id.json path:", relation2id_path)

        if not os.path.exists(node2id_path):
            raise FileNotFoundError(f"File not found: {node2id_path}")
        if not os.path.exists(entity_drug_path):
            raise FileNotFoundError(f"File not found: {entity_drug_path}")
        if not os.path.exists(relation2id_path):
            raise FileNotFoundError(f"File not found: {relation2id_path}")

        with open(node2id_path, 'r') as f:
            drug_set = json.load(f)
        with open(entity_drug_path, 'r') as f:
            entity_set = json.load(f)
        with open(relation2id_path, 'r') as f:
            relation_set = json.load(f)

        print("drug_set:", drug_set)

        for drug in drug_set:
            print("drug:", drug)
            id2entity[int(drug_set[drug])] = drug
        for ent in entity_set:
            id2entity[int(entity_set[ent])] = ent
        for rel in relation_set:
            id2relation[int(rel)] = rel

        self.id2entity = id2entity
        self.id2relation = id2relation

    def process_files_ddi(self, txt_file_path, output_csv_path):

        self.load_ent_id()

        data = []
        with open(txt_file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            h, t, r = int(triplet[0]), int(triplet[1]), int(triplet[2])
            drug1 = self.id2entity.get(h, 'Unknown')
            drug2 = self.id2entity.get(t, 'Unknown')
            relation = self.id2relation.get(r, 'Unknown')
            data.append([drug1, relation, drug2])
        print("Saving CSV to:", output_csv_path)

        with open(output_csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['d1', 'type', 'd2'])
            csv_writer.writerows(data)
        print(f"Data successfully converted to CSV and saved to {output_csv_path}")


# Example usage
task_dir = 'autodl-tmp/ImageMol-master/datasets/data/'
txt_file_path = 'datasets/data/S1_1/test_ddi.txt'
output_csv_path = 'datasets/data/S1_1/test_ddi.csv'
ddi_processor = DDIProcessor(task_dir)
ddi_processor.load_ent_id()
ddi_processor.process_files_ddi(txt_file_path, output_csv_path)
