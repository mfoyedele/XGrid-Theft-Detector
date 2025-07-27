import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # Use joblib for saving/loading models

# Load dataset
df = pd.read_csv('data/processed/extract_features_data.csv')

# Drop rows with missing target
df = df.dropna(subset=["FLAG"])

# Separate features and target
X = df.drop(columns=["FLAG"])
y = df["FLAG"]

# Fill missing values in features
X = X.fillna(0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Initialize and fit scaler on training data
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# Define the XGBoost model
classifier = xgb.XGBClassifier(
    subsample=0.8,
    scale_pos_weight=1,
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    gamma=0.3,
    colsample_bytree=0.6,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# Train the model
classifier.fit(X_train_scaled, y_train)

# Evaluate model performance before saving
from sklearn.metrics import classification_report
y_pred = classifier.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Save model and scaler to disk for later use
joblib.dump(classifier, 'xgrid-theft-prediction-model.pkl')
joblib.dump(sc, 'scaler.pkl')

print("Model and scaler saved successfully.")

