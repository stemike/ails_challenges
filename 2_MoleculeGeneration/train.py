import json
import torch
import numpy as np

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from modules.smiles_generator import SmilesGenerator
from modules.smiles_handler import SmilesHandler
from modules.smiles_dataset import SmilesDataset


def main():
    filename = "data/pp_smiles_{}.txt"
    data_train = SmilesDataset(filename.format("train"))
    data_val = SmilesDataset(filename.format("val"))
    train_model(data_train, data_val)


def train_model(data_train, data_val, batch_size=128, seed=0):
    loss_train = []
    loss_val = []
    num_batches_train = int(data_train.__len__() / batch_size)
    num_batches_val = int(data_val.__len__() / batch_size)

    smiles_handler = SmilesHandler()
    model = SmilesGenerator(smiles_handler.dict_len, padding_idx=smiles_handler.get_padding_index()).to(get_device())
    optimizer = Adam(model.parameters())
    # Raw predictions are expected
    loss_func = CrossEntropyLoss(ignore_index=smiles_handler.get_padding_index())

    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
    data_val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=4)

    for epoch in range(1, 15):
        epoch_loss_train = []
        model.train()

        print("Epoch {:03d}".format(epoch))
        print("===========")

        for batch_idx, samples in enumerate(data_train_loader):
            samples, targets = smiles_handler.split(samples)
            samples = samples.to(get_device())
            targets = targets.to(get_device())
            hidden = model.init_hidden(samples.shape[1])
            hidden = (hidden[0].to(get_device()), hidden[1].to(get_device()))

            optimizer.zero_grad()
            pred, hidden = model(samples, hidden)
            loss = loss_func(pred.view(-1, pred.shape[2]), targets.view(-1))
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                message = "-Epoch {:03d}: Batch {:04d} of {} with Training Loss of {}"
                print(message.format(epoch, batch_idx, num_batches_train, loss.item()))

            if batch_idx % 1000 == 0:
                torch.save(model.state_dict(), "{}/{:03d}.pt".format("output", epoch))
                print("Saved state to: {}/{:03d}.pt".format("output", epoch))

            epoch_loss_train.append(loss.item())

        loss_train.append(np.mean(epoch_loss_train))
        torch.save(model.state_dict(), "{}/{:03d}.pt".format("output", epoch))
        save_collection("output/lossListTrain.txt", loss_train)

        # after each epoch, evaluate model on validation set
        epoch_loss_val = []
        model.eval()
        with torch.no_grad():
            for batch_idx, samples in enumerate(data_val_loader):
                samples, targets = smiles_handler.split(samples)
                samples = samples.to(get_device())
                targets = targets.to(get_device())
                hidden = model.init_hidden(samples.shape[1])
                hidden = (hidden[0].to(get_device()), hidden[1].to(get_device()))

                pred, hidden = model(samples, hidden)

                loss = loss_func(pred.view(-1, pred.shape[2]), targets.view(-1)).item()

                if batch_idx % 100 == 0:
                    message = "-Epoch {:03d}: Validation Batch {:04d} of {} with Validation Loss of {}"
                    print(message.format(epoch, batch_idx, num_batches_val, loss))

                epoch_loss_val.append(loss)

            loss_val.append(np.mean(epoch_loss_val))
            print(f"Validation Loss for Epoch {epoch} of: {loss}")
            save_collection("output/lossListVal.txt", loss_val)


def get_device():
    # CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    return device


def save_collection(path, collection):
    with open(path, 'w') as f:
        json.dump(collection, f)


if __name__ == "__main__":
    main()
