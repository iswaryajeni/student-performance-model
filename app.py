from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl') 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = 1 if request.form['gender'] == 'Male' else 0
    studyhours = float(request.form['studyhours'])
    attendance = float(request.form['attendance'])

    prediction = model.predict([[age, gender, studyhours, attendance]])[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)