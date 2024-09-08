# summer-heat-waves-mobile-alert-system

pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
import pandas as pd
import matplotlib.pyplot as plt

# Load data
weather_data = pd.read_csv('historical_weather_data.csv')

# Basic analysis
print(weather_data.head())
print(weather_data.describe())

# Visualize temperature trends
plt.plot(weather_data['date'], weather_data['temperature'])
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Over Time')
plt.show()
weather_data['temp_moving_avg'] = weather_data['temperature'].rolling(window=7).mean()
weather_data['is_heat_wave'] = (weather_data['temperature'] > 35).astype(int)  # Example threshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Feature and target variables
X = weather_data[['temp_moving_avg', 'humidity', 'wind_speed']]  # Example features
y = weather_data['is_heat_wave']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate model
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))
from flask import Flask, request, jsonify
import numpy as np

app = Flask(_name_)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['temp_moving_avg'], data['humidity'], data['wind_speed']]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'is_heat_wave': int(prediction[0])})

if _name_ == "_main_":
    app.run(debug=True)
