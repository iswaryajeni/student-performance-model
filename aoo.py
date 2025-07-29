from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('student_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = 0 if request.form['gender'] == 'Male' else 1
    hours = float(request.form['studyhours'])
    attendance = float(request.form['attendance'])

    pred = model.predict([[age, gender, hours, attendance]])[0]
    result = "Pass" if pred == 1 else "Fail"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
