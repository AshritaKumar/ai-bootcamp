import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd

# Simple beginner-friendly dataset
data = {
    "Weather": ["Sunny", "Sunny", "Rainy", "Rainy", "Sunny", "Rainy"],
    "Temperature": ["Hot", "Cool", "Hot", "Cool", "Hot", "Cool"],
    "PlayTennis": ["No", "Yes", "Yes", "Yes", "No", "Yes"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert categorical features to numeric codes
X = df[["Weather", "Temperature"]].apply(lambda col: col.astype("category").cat.codes)
y = df["PlayTennis"].astype("category").cat.codes

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=2, random_state=0)
clf.fit(X, y)

# Map codes back to readable labels
feature_names = ["Weather", "Temperature"]
class_names = df["PlayTennis"].astype("category").cat.categories.tolist()

# Plot tree
plt.figure(figsize=(8, 5))
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    proportion=True,
    impurity=False,
    fontsize=12
)
plt.title("Decision Tree: Should We Play Tennis?")
plt.show()

# --- Ask a question ---
# User's scenario
question = {"Weather": "Sunny", "Temperature": "Cool"}

# Convert to numeric format for model
question_df = pd.DataFrame([question])
question_encoded = question_df.apply(
    lambda col: col.astype("category").cat.set_categories(
        df[col.name].astype("category").cat.categories
    ).cat.codes
)

# Predict
prediction_code = clf.predict(question_encoded)[0]
prediction_label = class_names[prediction_code]

print(f"Question: Weather={question['Weather']}, Temperature={question['Temperature']}")
print(f"Model Prediction: {prediction_label}")
