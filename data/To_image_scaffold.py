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

                    folder_path = os.path.join('drugood_scaffold', str(domain_id), str(cls_label))
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    img_file_path = os.path.join(folder_path, f"{index}.png")

                    img.save(img_file_path)


file_path = '/root/hrd_pg/gpt-4v-distribution-shift/drugood_scaffold.txt'
read_and_process_file(file_path)
