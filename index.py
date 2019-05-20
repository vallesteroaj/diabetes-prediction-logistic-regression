# Diabetes Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split as tts
from scipy import stats
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r'C:\Users\Lenovo\Desktop\machineLearningFiles\data\healthCareData\diabetes.csv')
i = []

x = data.iloc[:, 0:8].values
y = data.iloc[:, 8].values

ss = StandardScaler()
x = ss.fit_transform(x)

xTrain, xTest, yTrain, yTest = tts(x, y, test_size=0.2, random_state=0)

# Logistic Regression Result
model = LogisticRegression()
model.fit(xTrain, yTrain)
yPred = model.predict(xTest)

# Evaluations
print('Confusion Matrix\n', confusion_matrix(yTest, yPred))
print('\nAccuracy: ', accuracy_score(yTest, yPred))
print('\nClassification Report\n', classification_report(yTest, yPred))
fpr, tpr, thresholds = roc_curve(yTest, yPred)
print('AUC: ', auc(fpr, tpr))