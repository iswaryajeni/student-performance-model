import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv('student_performance (1).csv')

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Performance'] = data['Performance'].map({'Fail': 0, 'Pass': 1})


X = data[['Age', 'Gender', 'StudyHours', 'Attendance']]
y = data['Performance']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'student_model.pkl')
