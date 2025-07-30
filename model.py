
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')  # Replace 'crop_data.csv' with your dataset file

# Split the data into features and labels
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)



pickle.dump(model, open("model.pkl", "wb"))


