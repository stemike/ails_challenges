import json
import math

import xgboost as xgb
from scipy import stats
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class MoleculeClassifier:

    def train(self, d_train, d_val, num_rounds=500, early_stopping_rounds=50, params=None):
        eval_list = [(d_train, 'train'), (d_val, 'eval')]

        # fit model to training data
        if params is None:
            #Inbalanced Dataset
            params = {'base_score': 0.2790679482285984, 'booster': 'gbtree', 'gamma': 6.908937254227938,
                      'learning_rate': 0.3, 'max_delta_step': 5.771402440203418, 'max_depth': 21,
                      'min_child_weight': 5.390735467669154, 'n_jobs': -1, 'objective': 'binary:logistic',
                      'reg_alpha': 0.7241478484039084, 'reg_lambda': 0.3888674304090679, 'scale_pos_weight': 1,
                      'subsample': 0.823598326947018, 'tree_method': 'auto'}


        print("Training started")
        self.xgbtree = xgb.train(params, d_train, evals=eval_list, early_stopping_rounds=early_stopping_rounds,
                                 num_boost_round=num_rounds, verbose_eval=10)
        print(self.xgbtree.best_score)
        self.xgbtree.save_model('output/{}.model'.format("classifier"))

    def predict(self, X):
        return self.xgbtree.predict(X)

    def optimize_parameters(self, X, y, n_iter=200, seed=0):
        print("starting search")
        scale = sum(y == 0) / sum(y == 1)
        xgb_model = xgb.XGBClassifier(random_state=seed)
        grid = {
            "scale_pos_weight": [scale, scale * 2, math.sqrt(scale)]
        }
        clf = RandomizedSearchCV(xgb_model,
                                 grid,
                                 n_iter=n_iter,
                                 scoring="roc_auc",
                                 verbose=3,
                                 n_jobs=-1,
                                 cv=3,
                                 refit=True,
                                 random_state=seed)
        clf.fit(X, y)
        print(clf.best_score_)
        print(clf.best_params_)
        self.xgbtree = clf.best_estimator_
        file = json.dumps(clf.best_params_)
        f = open("output/param_dict.json", "w")
        f.write(file)
        f.close()
