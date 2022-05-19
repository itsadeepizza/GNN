
from torch_geometric.datasets import MoleculeNet

# Load the ESOL dataset
data = MoleculeNet(root=".", name="ESOL")
print(data.num_features)
from rdkit import Chem

molecule = Chem.MolFromSmiles(data[0]["smiles"])
molecule.Draw()