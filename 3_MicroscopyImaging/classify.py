import glob
import os

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from modules.model import CellClassifier


def main():
    labels = ['A549', 'CACO-2', 'HEK 293', 'HeLa', 'MCF7', 'PC-3', 'RT4', 'U-2 OS', 'U-251 MG']
    model = CellClassifier(len(labels))
    model.load_state_dict(torch.load("output/model.pt"))
    data_set, names = get_data()
    classify_cells(model, data_set, names, labels)


def classify_cells(model, data_set, names, labels):
    model = model.to(get_device())
    model.eval()
    predictions = []
    for i in range(len(data_set)):
        image = data_set[i].to(get_device())
        output = model(image.view(-1, image.shape[0], image.shape[1], image.shape[2]))
        _, pred = torch.max(output, 1)
        predictions.append(labels[pred.to("cpu")])
        if i % 100 == 0:
            print(f"Classified {i} images")

    df = pd.DataFrame({"file_id": names,
                       "cell_line": predictions})
    df.to_csv("output/predictions.csv", index=False)
    print("\nFinished Classifying Images")


def get_data(path="data/images_test"):
    #From https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                             std=std)
    ])
    images_test = []
    names = []
    rgb = []
    count = 0
    for f in sorted(glob.glob(f"{path}/*")):
        rgb.append(Image.open(f))
        if len(rgb) == 3:
            count += 1
            if count % 500 == 0:
                print(f"Added Image {count}")
            image = Image.merge("RGB", (rgb[0], rgb[1], rgb[2]))
            images_test.append(transformations(image))
            names.append(os.path.basename(f).split("_")[0].lstrip("0"))
            rgb = []
    print(names)
    return images_test, names

def get_device():
    # CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    return device

if __name__ == '__main__':
    main()
