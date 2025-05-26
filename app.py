from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    hours = int(request.form['hours'])
    features = np.array([[age, hours]])
    result = model.predict(features)[0]
    return f'<h2>Predicted Income: {"Above 50K" if result else "50K or below"}</h2>'

if __name__ == '__main__':
    app.run(debug=True)
