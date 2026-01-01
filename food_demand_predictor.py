import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("canteen_data.csv")

# Features & Target
X = data[['Day', 'Weather', 'Exam_Day', 'Price', 'Food_Type']]
y = data['Demand']

categorical_cols = ['Day', 'Weather', 'Exam_Day', 'Food_Type']
numerical_cols = ['Price']

# -----------------------------
# Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)

# -----------------------------
# Model
# -----------------------------
rf_model = RandomForestClassifier(random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5, None],
    'classifier__min_samples_split': [2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cv_scores = cross_val_score(best_model, X, y, cv=5)
print("Cross Validation Accuracy:", np.mean(cv_scores))

# Feature Importance
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
importances = best_model.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print("\nTop Important Features:")
print(feature_importance_df.head())

# -----------------------------
# USER INPUT
# -----------------------------
print("\n--- Enter details to predict food demand ---")

# Show valid options
day_options = data['Day'].unique()
weather_options = data['Weather'].unique()
exam_options = data['Exam_Day'].unique()
food_options = data['Food_Type'].unique()

print(f"Days available: {list(day_options)}")
day = input("Enter Day: ").title()
while day not in day_options:
    day = input(f"Invalid! Enter Day from {list(day_options)}: ").title()

print(f"Weather options: {list(weather_options)}")
weather = input("Enter Weather: ").title()
while weather not in weather_options:
    weather = input(f"Invalid! Enter Weather from {list(weather_options)}: ").title()

print(f"Exam Day options: {list(exam_options)}")
exam_day = input("Is it Exam Day? (Yes/No): ").title()
while exam_day not in exam_options:
    exam_day = input(f"Invalid! Enter Exam Day from {list(exam_options)}: ").title()

price = input("Enter Price (number): ")
while not price.isdigit():
    price = input("Invalid! Enter a valid number for Price: ")
price = int(price)

print(f"Food Type options: {list(food_options)}")
food_type = input("Enter Food Type: ").title()
while food_type not in food_options:
    food_type = input(f"Invalid! Enter Food Type from {list(food_options)}: ").title()

# Prepare input dataframe
user_input = pd.DataFrame({
    'Day': [day],
    'Weather': [weather],
    'Exam_Day': [exam_day],
    'Price': [price],
    'Food_Type': [food_type]
})

# Predict
prediction = best_model.predict(user_input)[0]
probability = best_model.predict_proba(user_input)[0]

print("\nPredicted Demand:", prediction)
print("Prediction Probability (High, Low):", probability)
