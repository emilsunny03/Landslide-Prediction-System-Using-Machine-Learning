# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import xgboost as xgb

# Load the dataset
df = pd.read_csv('augmented_synthetic_landslide_data_1000.csv')

# Preprocess the dataset
le = LabelEncoder()
df['Warning_Signal'] = le.fit_transform(df['Warning_Signal'])

# Feature Engineering: Create new features
df['Force'] = df['Vibration_Intensity'] * df['Acceleration']
df['Soil_Vibration_Ratio'] = df['Soil_Quality'] / (df['Vibration_Intensity'] + 1e-10)
df['Soil_Acceleration_Ratio'] = df['Soil_Quality'] / (df['Acceleration'] + 1e-10)

# Additional features
df['Inclination_Force_Ratio'] = df['Inclination'] / (df['Force'] + 1e-10)

# Features and target
X = df[['Soil_Quality', 'Vibration_Intensity', 'Acceleration', 'Inclination', 'Force',
        'Soil_Vibration_Ratio', 'Soil_Acceleration_Ratio', 'Inclination_Force_Ratio']]
y = df['Warning_Signal']

# Normalize the feature data using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE followed by undersampling to balance the dominant class
smote = SMOTE(random_state=42)
under = RandomUnderSampler(sampling_strategy='majority')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
X_train_resampled, y_train_resampled = under.fit_resample(X_train_resampled, y_train_resampled)

# XGBoost model
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)

# Train the model
xgb_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"\nXGBoost Model Accuracy: {accuracy_xgb * 100:.2f}%")
print("\nXGBoost Model Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['None', 'Red', 'Yellow'], zero_division=0))

# Save the XGBoost model
joblib.dump(xgb_model, 'xgboost_model.pkl')

# Predict on new data using the XGBoost model
new_data = pd.DataFrame([[0.7, 2.5, 3.0, 30, 0.7*2.5, 0.7/(2.5+1e-10), 0.7/(3.0+1e-10), 30/(0.7*2.5 + 1e-10)]],
                        columns=['Soil_Quality', 'Vibration_Intensity', 'Acceleration', 'Inclination', 'Force',
                                 'Soil_Vibration_Ratio', 'Soil_Acceleration_Ratio', 'Inclination_Force_Ratio'])
new_data_scaled = scaler.transform(new_data)
prediction = xgb_model.predict(new_data_scaled)
predicted_signal = le.inverse_transform([int(prediction[0])])[0]

# Interpret the prediction and display warning message
if predicted_signal == 'Red':
    landslide_risk = "High risk of landslide! Immediate action required."
elif predicted_signal == 'Yellow':
    landslide_risk = "Medium risk of landslide. Monitor the situation closely."
else:
    landslide_risk = "No significant risk of landslide."

# Print the predicted warning signal and alert
print(f'Predicted Warning Signal (XGBoost Model): {predicted_signal} - {landslide_risk}')
