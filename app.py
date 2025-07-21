import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('/content/heart_failure_clinical_records_dataset.csv')

# Confirm column names (for debug)
print("Columns in dataset:", df.columns.tolist())

# Features and target (update 'output' if your target is named differently)
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model trained. Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to 'model.pkl'
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("model.pkl file saved successfully.")
