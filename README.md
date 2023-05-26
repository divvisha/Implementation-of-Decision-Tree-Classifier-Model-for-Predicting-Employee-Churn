# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook 

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: Divyashree B S
RegisterNumber:  212221040044

import pandas as pd
data=pd.read_csv("/content/Employee.csv")

print("data.head():")
data.head()

print("data.info():")
data.info()

print("isnull() and sum():")
data.isnull().sum()

print("data value counts():")
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

print("data.head() for salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()

print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("accuracy value:")
accuracy

print("data prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

data.head():

<img width="875" alt="ex6 op1" src="https://github.com/divvisha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/127508123/4e3fa42d-7411-45c0-ae6e-0f51fca6bd40">

data.info():

<img width="490" alt="ex6 op2" src="https://github.com/divvisha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/127508123/0eadd4f1-d1ab-4e23-8817-ba8bc09e7397">

isnull() and sum():

<img width="379" alt="ex6 op3" src="https://github.com/divvisha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/127508123/433027bf-0637-42dd-bcc6-39e933228989">

data value counts():

<img width="298" alt="ex6 op4" src="https://github.com/divvisha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/127508123/82756c5f-d592-434e-90db-bfad99d75346">

data.head() for salary:

<img width="865" alt="ex6 op5" src="https://github.com/divvisha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/127508123/4cdcd3df-c390-4e94-9b9d-e401a72e43d8">

x.head():

<img width="852" alt="ex6 op6" src="https://github.com/divvisha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/127508123/f73e9082-3b1a-42a5-ac9f-ca4a99543107">

Accuracy value:

<img width="352" alt="ex6 op7" src="https://github.com/divvisha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/127508123/8784663d-ba90-4a6f-b1c3-40a3ac90e782">

data prediction:

<img width="892" alt="ex6 op8" src="https://github.com/divvisha/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/127508123/de4025ae-7830-49b0-a307-a78bbf2e488e">



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
