from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()
    # read files
    submission = pd.read_csv(args.submission, index_col=0)
    target = None
    if args.target is not None:
        target = pd.read_csv(args.target, index_col=0)

    # calculate AUC per task
    auc_per_task = []
    for j in range(target.shape[1]):
        # extract data
        y_score = submission.iloc[:, j]
        y_true = target.iloc[:, j]
        idx = (y_true != (-1))
        auc_per_task.append(roc_auc_score(y_true[idx], y_score[idx]))
    avg_auc = np.mean(auc_per_task)
    print(avg_auc)
