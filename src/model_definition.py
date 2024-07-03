import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMClassifier

class ModelFunctions:
    def __init__(self):
        self.model = LGBMClassifier(learning_rate=0.1,
                                    n_estimators= 300,
                                    scale_pos_weight=1,
                                    colsample_bytree=0.25,
                                    verbose=-1,
                                    n_jobs=-1
                                    )
    
    def train_model(self, X, y):
        '''
        does the model fit (train), returning the trained model
        '''
        return self.model.fit(X.values.astype('float'), y.values.astype('int'))
    
    def class_threshold(self, clf,  X_pred, threshold = 0.199):

        '''
        classifies a customer based on the threshold
        '''

        if isinstance(X_pred, np.ndarray):
            X_pred = X_pred.astype('float')
        else:
            X_pred = X_pred.values.astype('float')
        
        y_proba = clf.predict_proba(X_pred)[:, 1]
        y_pred = np.where(y_proba >= threshold, 1, 0)

        return y_pred