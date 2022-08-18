import numpy as np
import torch
from rdkit import Chem

from modules.smiles_generator import SmilesGenerator
from modules.smiles_handler import SmilesHandler


def main():
    handler = SmilesHandler()
    model = SmilesGenerator(handler.dict_len, padding_idx=handler.get_padding_index())
    model.load_state_dict(torch.load("output/model.pt"))
    sample_molecules(model, handler)


def sample_molecules(model, smiles_handler, n_mol=11000, seed=0):
    model = model.to(get_device())
    model.eval()

    molecule_count = 0
    while molecule_count <= n_mol:
        molecule = []
        index = smiles_handler.get_start_index()
        stop_index = smiles_handler.get_stop_index()

        hidden = model.init_hidden(1)
        hidden = (hidden[0].to(get_device()), hidden[1].to(get_device()))
        while index != stop_index:
            x = torch.tensor(index).long()
            x = x.reshape(1, 1)
            x = x.to(torch.device(get_device()))

            output, hidden = model(x, hidden)
            index = get_mol(output.to("cpu"))
            if index == stop_index and len(molecule) == 0:
                continue
            molecule.append(index)
        mol_string = smiles_handler.decode_smile(molecule)
        if is_mol_valid(mol_string[:-1]):
            with open("output/molecules.txt", "a") as file:
                file.write(mol_string)
            molecule_count += 1
            if molecule_count % 10 == 0:
                print(f"Generated {molecule_count} out of {n_mol}")
        else:
            print(f"\tInvalid Molecule: {mol_string}")
    print("\nFinished Generating Molecules")


def is_mol_valid(molecule):
    return Chem.MolFromSmiles(molecule) is not None


def get_mol(output):
    """
    Extract molecule from distribution
    :param output: the prediction of a classifier
    :return: an index for a molecule
    """
    probabilities = torch.nn.functional.softmax(output, dim=2).data[0][0]
    return torch.multinomial(probabilities, 1)[0]


def get_device():
    # CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    return device


if __name__ == '__main__':
    main()
