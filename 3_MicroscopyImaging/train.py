import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from modules.model import CellClassifier


def main():
    data_dir = "data"
    #From https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = {
        'train_balanced': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((0, 90)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'val_balanced': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ["train_balanced", "val_balanced"]}

    #show_transformations(image_datasets["train_balanced"], mean, std)

    train_model(image_datasets["train_balanced"], image_datasets["val_balanced"])


def train_model(data_train, data_val, batch_size=64, output_dir="output"):
    loss_train = []
    loss_val = []
    accuracy_val = []
    num_batches_train = int(data_train.__len__() / batch_size)
    num_batches_val = int(data_val.__len__() / batch_size)
    data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
    data_loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=4)

    model = CellClassifier(len(data_train.classes)).to(get_device())
    optimizer = Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    loss_func = CrossEntropyLoss()

    for epoch in range(1, 100):
        epoch_loss_train = []
        model.train()

        print(f"Epoch {epoch:03d}")
        print("===========")

        for batch_idx, (samples, targets) in enumerate(data_loader_train):
            samples = samples.to(get_device())
            targets = targets.to(get_device())

            optimizer.zero_grad()
            y_pred = model(samples)
            loss = loss_func(y_pred, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                message = "-Epoch {:03d}: Batch {:04d} of {} with Training Loss of {}"
                print(message.format(epoch, batch_idx, num_batches_train, loss.item()))

            if batch_idx % 15 == 0:
                torch.save(model.state_dict(), f"{output_dir}/{epoch:03d}.pt")
                print(f"Saved state to: {output_dir}/{epoch:03d}.pt")

            epoch_loss_train.append(loss.item())

        if epoch >= 50:
            exp_lr_scheduler.step()

        loss_train.append(np.mean(epoch_loss_train))
        torch.save(model.state_dict(), f"{output_dir}/{epoch:03d}.pt")
        save_collection("output/lossListTrain.txt", loss_train)

        # after each epoch, evaluate model on validation set
        epoch_loss_val = []
        epoch_acc_val = []
        model.eval()
        with torch.no_grad():
            for batch_idx, (samples, targets) in enumerate(data_loader_val):
                samples = samples.to(get_device())
                targets = targets.to(get_device())

                y_pred = model(samples)
                loss = loss_func(y_pred, targets).item()
                epoch_acc_val.append(accuracy_score(targets.cpu(), torch.max(y_pred, 1)[1].cpu()))

                if batch_idx % 10 == 0:
                    message = "-Epoch {:03d}: Validation Batch {:04d} of {} with Validation Loss of {}"
                    print(message.format(epoch, batch_idx, num_batches_val, loss))

                epoch_loss_val.append(loss)

            loss_val.append(np.mean(epoch_loss_val))
            accuracy_val.append(np.mean(epoch_acc_val))
            print(f"Validation Loss for Epoch {epoch} of: {np.mean(epoch_loss_val)}")
            print(f"Validation Accuracy for Epoch {epoch} of: {np.mean(epoch_acc_val)}")
            save_collection("output/lossListVal.txt", loss_val)
            save_collection("output/AccVal.txt", accuracy_val)


def show_transformations(data_set, mean, std):
    #From: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    data_loader = DataLoader(data_set, batch_size=5, shuffle=True, num_workers=4)

    inputs, classes = next(iter(data_loader))
    title = [data_set.classes[x] for x in classes]

    inp = torchvision.utils.make_grid(inputs).numpy().transpose((1, 2, 0))

    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()


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
