# EXP-07 : Implementation of Decision Tree Regressor Model for Predicting the Salary of the Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import pandas and read the csv file.
2. Encoding the data and Import Decision tree classifier.
3. Fit the data in the model.
4. Find the MSE , R^2 and the Predicted values.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: J.JENISHA
RegisterNumber:  212222230056
```
```python
import pandas as pd
df=pd.read_csv("Salary.csv")

df.head()
df.info()
df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
df.head()

x=df[["Position","Level"]]
y=df["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
print("Mean Squared Error : ",mse)

r2=metrics.r2_score(y_test,y_pred)
print("R Squared Error : ",r2)

dt.predict([[5,6]])
```

## Output:
#### df.head()
![Screenshot 2024-05-05 124845](https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119405070/5744f81f-c51e-44a4-8b0c-e7f0437c5f9a)

#### df.info()
<img src="https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119405070/0cd7a1f6-e835-43e3-b805-951ef92488f5" height=200 width=300>

#### Null values
![Screenshot 2024-05-05 125001](https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119405070/5820ae76-0735-4618-8c3b-7f183f3e5b95)

#### Labeled data
![Screenshot 2024-05-05 125112](https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119405070/8c25c8a8-1d7f-4f71-b113-699f3adb0cb2)

#### MSE
![Screenshot 2024-05-05 125145](https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119405070/f0534556-c913-4a7f-b073-2889e3d2eebb)

#### RSE
![Screenshot 2024-05-05 125211](https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119405070/658d2214-3854-4dc6-9f6c-22d3eb2cc22b)

#### Predicted value
![Screenshot 2024-05-05 125322](https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119405070/92e2922f-b999-4303-973b-b8a55245f298)
![Screenshot 2024-05-05 125343](https://github.com/Jenishajustin/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119405070/77da0ac1-2330-4ccb-895d-bf40a7756b86)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
