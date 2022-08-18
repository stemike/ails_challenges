import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import BorderlineSMOTE
from joblib import dump, load
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score, plot_precision_recall_curve, \
    plot_roc_curve
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV


def main():
    train_classifiers = True
    predict_labels = True

    data_path = "data/{}"
    out_path = "output/{}/"
    y_cols = ["Task1", "Task2", "Task3", "Task4", "Task5", "Task6", "Task7", "Task8", "Task9"]

    print("Program started")

    if train_classifiers:
        auc_list = []
        df_train = pd.read_csv(data_path.format("desc_train.csv"))

        for y_col in y_cols:
            print(y_col)
            intermediate_df = df_train[df_train[y_col].notna()]
            X = intermediate_df.drop(y_cols, axis=1).values
            y = intermediate_df[y_col].astype(int).values
            auc_list.append(train(X, y, out_path.format(y_col)))
        print("Mean ROC: {}".format(np.mean(auc_list)))

    if predict_labels:
        X_test = pd.read_csv(data_path.format("desc_test.csv")).values

        prediction_list = []
        for y_col in y_cols:
            clf = load(out_path.format(y_col) + "classifier_refit.joblib")
            predictions = clf.predict_proba(X_test)[:, 1]
            prediction_list.append(pd.Series(predictions, name=y_col))

        df = pd.DataFrame(prediction_list).T
        print(df.head())
        df.to_csv("output/submission.csv")


def train(X, y, out_path, seed=0):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    sm = BorderlineSMOTE(random_state=seed)
    X_train, y_train = sm.fit_sample(X_train, y_train)
    eval_set = [(X_train, y_train), (X_val, y_val)]

    #return find_params(X_train, y_train, X_val, y_val, out_path, seed)

    with open(out_path + "param_dict.json") as f:
        st = f.read()
        params = json.loads(st)

    model = xgb.XGBClassifier(n_estimators=2000, **params)

    model.fit(X_train, y_train, early_stopping_rounds=40, eval_set=eval_set, eval_metric=["auc"])
    dump(model, out_path + "classifier.joblib")
    auc = get_performance(X_val, y_val, model, out_path)

    X = np.append(X_train, X_val, axis=0)
    y = np.append(y_train, y_val, axis=0)

    final_model = xgb.XGBClassifier(n_estimators=model.best_iteration, **params)
    final_model.fit(X, y)
    dump(final_model, out_path + "classifier_refit.joblib")

    return auc


def find_params(X_train, y_train, X_val, y_val, out_path, seed=0):
    ratio = sum(y_train == 0) / sum(y_train == 1)

    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=seed, n_estimators=50)
    grid = {
        "learning_rate": [0.06],
        "colsample_bytree": [0.8, 0.9],
        "subsample": [0.8, 0.9],
        "max_depth": [5, 7, 9],
        "min_child_weight": [5, 7, 8],
        "gamma": [0.1, 0.3, 0.5],
        "objective": ['binary:logistic']
    }

    print("starting search")

    clf = GridSearchCV(xgb_model, grid, scoring="roc_auc", verbose=1, n_jobs=-1, cv=3, refit=True)

    clf.fit(X_train, y_train)
    print(clf.best_score_)
    print(clf.best_params_)
    file = json.dumps(clf.best_params_)
    with open(out_path + "param_dict.json", "w") as f:
        json.dump(clf.best_params_, f)
    return get_performance(X_val, y_val, clf, out_path)


def sensitivity_specificity(y, probabilities, out_path):
    bins = np.array(range(0, 101, 5)) / 100
    for i in [0, 1]:
        if i == 0:
            color = "red"
            label = "inactive"
        else:
            color = "green"
            label = "active"
        plt.hist(probabilities[y == i], color=color, bins=bins, label=label)
        plt.xticks(bins, bins, rotation=90)
        plt.title("Sensitivity-Specificity Plot - AUC: {}".format(roc_auc_score(y, probabilities) * 100))
        plt.legend()
        plt.savefig(out_path + "sensitivity_specificity_{}".format(i))


def get_performance(X, y_true, model, out_path):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    average_precision = average_precision_score(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)
    print("ROC Auc: {:2}".format(roc_auc * 100))
    print(classification_report(y_true, predictions))
    sensitivity_specificity(y_true, probabilities, out_path)

    roc_curve = plot_roc_curve(model, X, y_true)
    roc_curve.ax_.set_title("2-class ROC curve")
    plt.savefig(out_path + "ROC.png")

    pr_re_curve = plot_precision_recall_curve(model, X, y_true)
    pr_re_curve.ax_.set_title("2-class Precision-Recall curve: "
                              "AP={0:0.2f}".format(average_precision))
    plt.savefig(out_path + "PRC.png")
    return roc_auc


if __name__ == '__main__':
    main()
