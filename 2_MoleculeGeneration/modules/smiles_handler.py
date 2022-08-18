import json

import torch
import numpy as np


class SmilesHandler:
    def __init__(self):
        """
        sys_characters = ["Padding", "Start", "\n"]
        elements = ["Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au", "B", "Ba", "Be", "Bh", "Bi", "Bk", "Br", "C",
                    "Ca", "Cd", "Ce", "Cf", "Cl", "Cm", "Co", "Cr", "Cs", "Cu", "Db", "Dy", "Er", "Es", "Eu", "F",
                    "Fe", "Fm", "Fr", "Ga", "Gd", "Ge", "H", "He", "Hf", "Hg", "Ho", "Hs", "I", "In", "Ir", "K", "Kr",
                    "La", "Li", "Lr", "Lu", "Md", "Mg", "Mn", "Mo", "Mt", "N", "Na", "Nb", "Nd", "Ne", "Ni", "No", "Np",
                    "O", "Os", "P", "Pa", "Pb", "Pd", "Pm", "Po", "Pr", "Pt", "Pu", "Ra", "Rb", "Re", "Rf", "Rh", "Rn",
                    "Ru", "S", "Sb", "Sc", "Se", "Sg", "Si", "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl",
                    "Tm", "U", "V", "W", "Xe", "Y", "Yb", "Zn", "Zr"]
        elements_lower = [e.lower() for e in elements]
        misc_characters = ["+", "-", ".", "=", "#", "$", ":", "/", "\\", "%", "[", "]",
                           "(", ")", "@", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        all_chars = sys_characters + elements + elements_lower + misc_characters
        self.char2int = dict(zip(all_chars, range(len(all_chars))))
        self.int2char = dict(zip(range(len(all_chars)), all_chars))
        """
        with open("data/int2char.txt", 'r') as f:
            self.int2char = json.load(f)
            self.int2char = {int(k): v for k, v in self.int2char.items()}
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.dict_len = len(self.int2char)

    def encode_smile(self, smile):
        smile_len = len(smile)
        indices = []
        i = 0
        while i < smile_len:
            if i + 1 < smile_len and smile[i: i + 2] in self.char2int.keys():
                index = self.char2int[smile[i: i + 2]]
                i += 1
            else:
                index = self.char2int[smile[i]]
            i += 1
            indices.append(index)
        return torch.tensor(indices)

    def decode_smile(self, indices):
        return "".join([self.int2char[int(i)] for i in indices])

    def split(self, batch):
        """
        Padds a batch of sequences and extracts their labels
        :param batch: a batch of sequences
        :return: a padded batch of sequences and their labels
        """
        samples = []
        targets = []
        for sample in batch:
            indices = self.encode_smile(sample)
            sample = np.insert(indices, 0, self.get_start_index())
            target = torch.tensor(np.append(indices, self.get_stop_index()))
            samples.append(sample)
            targets.append(target)
        samples = torch.nn.utils.rnn.pad_sequence(samples, padding_value=self.get_padding_index())
        targets = torch.nn.utils.rnn.pad_sequence(targets, padding_value=self.get_padding_index())
        # shape(seq, batch, features)
        return samples, targets

    def get_padding_index(self):
        return self.char2int["Padding"]

    def get_start_index(self):
        return self.char2int["Start"]

    def get_stop_index(self):
        return self.char2int["\n"]
