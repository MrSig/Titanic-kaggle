import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


                
models = [LogisticRegression(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), XGBClassifier()]
param_grids =[
    {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-4, 4, 10),
        'solver': ['liblinear']
    },
    {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
    },
    {
        'max_depth': np.linspace(1, 32, 32, endpoint=True),
        'min_samples_split': np.linspace(0.1, 1.0, 10, endpoint=True),
        'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True)
    },
    {
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [1, 5, 10, 25],
        'min_samples_split': [10, 12, 16, 18],
        'n_estimators': [100, 700, 1500]
    },
    {
        'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
        'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
    }
]


class Model:
    def __init__(self, model=None):                        
        self.model = model
        self.name = type(self.model).__name__
    
    def train_best(self, model, x_dat, y_dat, params, cv):
        grid = GridSearchCV(model, param_grid=params, cv=cv)
        grid.fit(x_dat,y_dat)
        return grid.best_estimator_
    
    def score(self, x_dat, y_dat):
        return cross_val_score(self.model, x_dat, y_dat, cv)
    
    def predict(self,x_to_pred):
        y_pred = self.model.predict(x_to_pred)
        return y_pred
    
    def save(self, path):
        pickle.dump(self.model, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.name, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)
        
    @staticmethod
    def load(path):
        model_inst = Model()
        model_inst.model = pickle.load(open(path, 'rb'))
        model_inst.name = pickle.load(open(path, 'rb'))
        return model_inst