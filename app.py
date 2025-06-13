from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]
    return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction:.2f}K')

if __name__ == '__main__':
    app.run(debug=True)