import pandas as pd
data=pd.read_csv("Dataset/student.csv")
print(data)
print(data.head())
print(data.isnull().sum())
print(data.isnull().sum().sum())
data=data.dropna(subset=["result"])
numerical_columns=["study_hours","attendance","previous_marks"]
for column in numerical_columns:
    data[column]=data[column].fillna(data[column].median())
print(data)
print(data.isnull().sum())
print(data.duplicated().sum())
data=data.drop_duplicates()
print(data)
data.to_csv("dataset/students_clean.csv",index=False)
print("data saved")
data["result"]=data["result"].map({'Pass':1,'Fail':0})
print(data)
x=data[["study_hours","attendance","previous_marks"]]
y=data["result"]
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
print("model training completed")
y_pred=model.predict(x_test)
print(y_test.values)
print(y_pred)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
accuracy=accuracy_score(y_test,y_pred)
print(accuracy*100,"%")
print(classification_report(y_test,y_pred,zero_division=0))
import joblib
joblib.dump(model,"models/student_models.pkl")
print("success")