# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:NITHYAA SRI S S 
RegisterNumber:212222230100  
*/

import pandas as pd
data=pd.read_csv("/content/Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier,plot_tree
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plot_tree(dt, feature_names=x.columns, class_names=['not left', 'left'], filled=True)
plt.show()

```

## Output:
![decision tree classifier model](sam.png)
# DATASET:
![01](https://github.com/ssnithyaasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119122478/f1f86d7c-0d2b-4f24-8f18-71db998d9f75)
# ACCURACY AND dt predict()
![02](https://github.com/ssnithyaasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119122478/a5ec0397-d5e8-4900-a384-b8fa2736aee6)

![03](https://github.com/ssnithyaasri/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119122478/0e69fa7f-79e8-457b-9e76-48b657c229c4)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
