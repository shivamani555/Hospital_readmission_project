import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# load dataset
data = pd.read_excel("hospital_readmission_dataset.xlsx")

# split features and target
X = data.drop("readmitted", axis=1)
y = data["readmitted"]

# scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model trained and saved")