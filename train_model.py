import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv('train.csv')

X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained using CSV and saved!")