import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
import pickle


df = pd.read_csv('weather\weather_clouds.csv')
df.head()
days = df['Day']
labels = {'Low':0, 'High':1} 
output = []

# for i in range(0,len(output_data)):
#     if output_data[i] <= 10:
#         output.append(0)
#     else:
#         output.append(1)

corr = df.corr()
plt.figure(figsize=(13, 13))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.show()
df.info()
df['Global Horizontal UV Irradiance (280-400nm)'] = df['Global Horizontal UV Irradiance (280-400nm)'].apply(lambda x: 0 if x <= 10 else 1)
X = df.drop(columns=['Year','Month','Day','Minute','Dew Point','Pressure','Wind Speed','Precipitable Water','Global Horizontal UV Irradiance (280-400nm)'], axis=0)
print(X.head())
y = df['Global Horizontal UV Irradiance (280-400nm)']

# print(X.shape)
# print(y.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)


lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
#filename = 'finalized_model.sav'
#pickle.dump(lr, open(filename, 'wb'))
# Evaluation metrics
y_pred_lr = lr.predict(X_test)
log_train = round(lr.score(X_train, y_train) * 100, 2)
log_accuracy = round(accuracy_score(y_pred_lr, y_test) * 100, 2)

#From HW1
print("Training Accuracy    :",log_train ,"%")
print("Model Accuracy Score :",log_accuracy ,"%")
print("\033[1m--------------------------------------------------------\033[0m")
print("Classification_Report: \n",classification_report(y_test,y_pred_lr,zero_division=0))
print("\033[1m--------------------------------------------------------\033[0m")

# con_m = confusion_matrix(y_test, y_pred_lr)
# temp = ConfusionMatrixDisplay(confusion_matrix=con_m)
# temp.plot()
# plt.show()

# power_days = 0
# est_power_days = 0
# day_count = 32
# for i in range(0,len(y_pred_lr)-4):
#     if y_pred_lr[i] == 1 and y_pred_lr[i+1] == 1 and y_pred_lr[i+2] == 1 and y_pred_lr[i+3] == 1 and days[i] != day_count:
#         est_power_days += 1
#         day_count = days[i]
# print(est_power_days)

# day_count = 32
# for i in range(0,len(y)-4):
#     if y[i] == 1 and y[i+1] == 1 and y[i+2] == 1 and y[i+3] == 1 and days[i] != day_count:
#         power_days += 1
#         day_count = days[i]
# print(power_days)

for i in y_pred_lr:
    print(i)