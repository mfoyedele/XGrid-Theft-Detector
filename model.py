# Importing essential libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle

# Loading the dataset
df = pd.read_csv('/data/features/selected_features.csv')

# Model Building
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

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Define the XGBoost model
classifier = xgb.XGBClassifier(
    subsample=0.8,
    scale_pos_weight=np.float64(10.720954356846473),
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    gamma=0,
    colsample_bytree=0.8,
    use_label_encoder=False,  
    eval_metric='logloss'     
)


classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'xgrid-theft-prediction-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))