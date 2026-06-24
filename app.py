from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])

        features = np.array([[area, bedrooms, bathrooms]])
        prediction = model.predict(features)

        result = f"💰 Estimated Price: ₹ {int(prediction[0])}"

    except Exception as e:
        result = "❌ Invalid input! Please enter correct numbers."

    return render_template('index.html', result=result)

if __name__== "__main__":
    app.run(debug=True)