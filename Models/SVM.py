# Effective for high dimentional spaces (More attributes)

import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

class SVM:
    def train(self, file_name):
        data = pd.read_csv(file_name)
        data = pd.read_csv('Track_1.xls')
        col = list(data.columns)
        attributes = col[1:21]

        X = data[attributes].values
        y = data[col[21]].values.astype('int')

        le = LabelEncoder()
        for i in range(len(X[0])):
            X[:,i] = le.fit_transform(X[:,i])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = svm.SVC()
        # model = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
        model.fit(X_train, y_train.astype('int'))

        predictions = model.predict(X_test.astype('int'))
        acc = accuracy_score(y_test.astype('int'), predictions)
        print("Accuracy: ", acc)
        
        self.model_name = 'SVM.sav'
        pickle.dump(model, open(self.model_name, 'wb'))
        
    def predict(self, file_name):
        data = pd.read_csv(file_name)
        data = pd.read_csv('Track_1.xls')
        col = list(data.columns)
        attributes = col[1:21]

        X = data[attributes].values
        y = data[col[21]].values.astype('int')

        le = LabelEncoder()
        for i in range(len(X[0])):
            X[:,i] = le.fit_transform(X[:,i])
        
        model = pickle.load(open(self.model_name, 'rb'))
        predictions = model.predict(X.astype('int'))
        ac = accuracy_score(predictions, y)
        print ('Accuracy: ', ac)
        print (predictions)
        
# model = SVM()

# model.train('Track_1.xls')
# model.predict('Track_1.xls')