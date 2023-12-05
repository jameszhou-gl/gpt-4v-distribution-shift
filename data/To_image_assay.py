import os
from rdkit import Chem
from rdkit.Chem import Draw

def read_and_process_file(file_path):
    with open(file_path, 'r') as file:
        index = 0
        for line in file:
            index = index + 1
            if line.strip():


                smiles, cls_label, domain_id = line.split()


                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img = Draw.MolToImage(mol, (224, 224))

                    if cls_label == '0':
                        cls_label = 'inactive'
                    if cls_label == '1':
                        cls_label = 'active'
                    folder_path = os.path.join('datasets/drugood_assay2', str(domain_id), cls_label)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)


                    img_file_path = os.path.join(folder_path, f"{index}.png")


                    img.save(img_file_path)



file_path = '/Users/herundong/PycharmProjects/pythonProject/gpt-4v-distribution-shift/drugood_assay.txt'
read_and_process_file(file_path)
