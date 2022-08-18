from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()
    # read files
    submission = pd.read_csv(args.submission)
    target = None
    if args.target is not None:
        target = pd.read_csv(args.target)
    merged = pd.merge(target, submission, on="file_id", how='left', suffixes=('_true', '_pred'))
    # encode class labels
    classes = ["A549", "CACO-2", "HEK 293", "HeLa", "MCF7", "PC-3", "RT4", "U-2 OS", "U-251 MG"]
    merged = merged.replace(classes, list(range(9)))
    # calculate bacc
    print(balanced_accuracy_score(merged.cell_line_true, merged.cell_line_pred))