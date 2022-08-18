import pandas as pd
import xgboost as xgb


class DataLoader:
    def __init__(self, path, load_pandas=True, is_test_set=False, drop_na=False):
        if load_pandas:
            df = pd.read_csv(path + ".csv")
            if is_test_set:
                X = df.drop(["Smiles"], axis=1)
                y = None
                dmatrix = xgb.DMatrix(X.values)
            else:
                if drop_na:
                    df = df.drop_na()
                X = df.drop(["Smiles", "Activity"], axis=1)
                y = df["Activity"]
                dmatrix = xgb.DMatrix(X.values, y.values)
            dmatrix.save_binary(path + ".buffer")
        else:
            dmatrix = xgb.DMatrix(path + ".buffer")
            X = None
            y = dmatrix.get_label()

        self.X = X
        self.y = y
        self.dmatrix = dmatrix

    def get_dmatrix(self):
        return self.dmatrix

    def get_X(self):
        return self.X.values

    def get_y(self):
        return self.y.values

    def get_split_data(self):
        return self.get_X(), self.get_y()
