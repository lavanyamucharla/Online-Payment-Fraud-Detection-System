import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset (replace 'path/to/your/dataset.csv' with the actual file path)
df = pd.read_csv(r"C:\Users\prade\Downloads\PS_20174392719_1491204439457_log.csv (2)\PS_20174392719_1491204439457_log.csv")

# Step 1: Data exploration and preprocessing
# Drop unnecessary columns (assuming 'nameOrig' and 'nameDest' are irrelevant)
df.drop(columns=['nameOrig', 'nameDest','isFlaggedFraud'], inplace=True)

# Encode 'type' column (transaction type)
encoder = LabelEncoder()
df['type'] = encoder.fit_transform(df['type'])

# Step 2: Define features (X) and target (y)
X = df.drop(columns=['isFraud'])  # Features: all columns except 'isFraud'
y = df['isFraud']  # Target: 'isFraud' column

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 4: Train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)

# Step 5: Evaluate the model (optional)
# You can check the accuracy here if desired
from sklearn.metrics import accuracy_score
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy on test data:", accuracy)

# Step 6: Save the trained model and encoder
joblib.dump(dt_model, "decision_tree_model.joblib")
joblib.dump(encoder, "encoder.joblib")
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Compute sample weights to handle class imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Train the decision tree model with the sample weights
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train, sample_weight=sample_weights)

print("Model and encoder saved successfully.")
