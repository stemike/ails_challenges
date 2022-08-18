import json
from math import sqrt

import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer, confusion_matrix, recall_score, \
    classification_report, precision_score, average_precision_score, plot_precision_recall_curve
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
import numpy as np
from joblib import dump, load
from rdkit import Chem
from sklearn.model_selection import KFold, GridSearchCV, RepeatedStratifiedKFold, cross_val_score, RandomizedSearchCV

from model import MoleculeClassifier
from data_provider import DataLoader


def main():
    print("Program started")
    # find_params()
    # train()
    predict_test_labels()


def train(path="data_corona/{}", evaluate=False, seed=1):
    id = "post_grid"
    #    dl_train = DataLoader(path.format("desc_train_set"))
    #    dl_val = DataLoader(path.format("desc_validation_set"))

    X_train = pd.read_csv(path.format("X_train_smote.csv"))
    y_train = pd.read_csv(path.format("Y_train_smote.csv"))["Activity"]
    X_val = pd.read_csv(path.format("X_val_smote.csv"))
    y_val = pd.read_csv(path.format("Y_val_smote.csv"))["Activity"]

    params = {
        "scale_pos_weight": np.sqrt(sum(y_train == 0) / sum(y_train == 1)),
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_weight": 1,
        "reg_alpha": 0.75,
        "reg_lambda": 0.50,
        "gamma": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_delta_step": 1,
        "random_state": seed
    }

    model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=150, **params)

    eval_set = [(X_train, y_train), (X_val, y_val)]

    model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=eval_set, eval_metric=["auc", "aucpr"])
    dump(model, "output/classifier_{}.joblib".format(id))
    get_preformance(X_val, y_val, model, id)
    X = X_train.append(X_val, ignore_index=True)
    y = y_train.append(y_val, ignore_index=True)
    params["scale_pos_weight"] = 2 * np.sqrt(sum(y == 0) / sum(y == 1))
    final_model = xgb.XGBClassifier(n_estimators=model.best_iteration, **params)
    final_model.fit(X, y)
    dump(final_model, "output/classifier_{}_Abgabe.joblib".format(id))


def predict_test_labels(path="data_corona/{}", output="output/submission.csv"):
    #dl_test = DataLoader(path.format("X_test_smote"), is_test_set=True)

    X_test = pd.read_csv(path.format("X_test_smote.csv"))
    # make prediction
    clf = load("output/classifier_post_grid_Abgabe.joblib")
    predictions = clf.predict_proba(X_test)[:, 1]
    df = pd.DataFrame(predictions)
    df = df.rename(columns={0: "", 1: "Activity"})
    df.to_csv(output)


def find_params(n_iter=50, path="data_corona/{}", seed=0):
    # dl_train = DataLoader(path)

    X_train = pd.read_csv(path.format("X_train_smote.csv"))
    y_train = pd.read_csv(path.format("Y_train_smote.csv"))["Activity"]
    ratio = sum(y_train == 0) / sum(y_train == 1)

    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=seed, n_estimators=50)
    grid = {
        "scale_pos_weight": [np.sqrt(ratio)],
        "learning_rate": [0.3],
        "max_depth": [7],
        "min_child_weight": [1],
        "reg_alpha": [0.75],
        "reg_lambda": [0.50],
        "gamma": [0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "max_delta_step": range(11)
    }

    print("starting search")

    clf = RandomizedSearchCV(xgb_model, grid, n_iter=n_iter, scoring="roc_auc", verbose=3, n_jobs=-1, cv=3, refit=True,
                             random_state=seed)
    clf = GridSearchCV(xgb_model, grid, scoring="roc_auc", verbose=3, n_jobs=-1, cv=3, refit=True)

    clf.fit(X_train, y_train)
    print(clf.best_score_)
    print(clf.best_params_)
    xgbtree = clf.best_estimator_
    file = json.dumps(clf.best_params_)
    f = open("output/param_dict.json", "w")
    f.write(file)
    f.close()
    X_val = pd.read_csv(path.format("X_val_smote.csv"))
    y_val = pd.read_csv(path.format("Y_val_smote.csv"))["Activity"]

    get_preformance(X_val, y_val, clf, "gridesearch")


def sensitivity_specificity(y, probabilities, id=""):
    bins = np.array(range(0, 101, 5)) / 100
    for i in [0, 1]:
        if i == 0:
            color = "red"
            label = "inactive"
        else:
            color = "green"
            label = "active"
        plt.hist(probabilities[y == i], color=color, bins=bins, density=True, label=label)
        plt.xticks(bins, bins, rotation=90)
        plt.title("Sensitivity-Specificity Plot - AUC: {}".format(roc_auc_score(y, probabilities) * 100))
        plt.legend()
        plt.savefig("output/sensitivity_specificity_{}{}".format(id, i))


def get_label_from_prob(probabilities, threshold=0.5):
    return [int(prob >= threshold) for prob in probabilities]


def get_preformance(X, y_true, model, id):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    average_precision = average_precision_score(y_true, probabilities)
    print("ROC Auc: {:2}".format(roc_auc_score(y_true, probabilities) * 100))
    print(classification_report(y_true, predictions))
    sensitivity_specificity(y_true, probabilities, id=id)

    disp = plot_precision_recall_curve(model, X, y_true)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(average_precision))
    plt.savefig("output/PRC_{}.png".format(id))


if __name__ == '__main__':
    main()
