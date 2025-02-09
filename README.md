# Heart-Disease-Prediction-Project

## Overview
This project aims to predict heart disease using machine learning techniques. The dataset is split into training and testing sets to train a logistic regression model that classifies whether a patient is likely to have heart disease based on various health indicators.

## Technologies Used
- **Python**: Programming language used for implementation.
- **Machine Learning**: Applied for predictive modeling.
- **Logistic Regression**: The algorithm used for classification.
- **Train-Test Split**: Used to split data for training and evaluation.

## Dataset
The dataset consists of medical attributes such as age, cholesterol levels, blood pressure, and more. Each instance is labeled with the presence or absence of heart disease.

## Usage
Run the Python script to train and evaluate the model:
```bash
python heart_disease_prediction.py
```

## Implementation
1. Load the dataset.
2. Preprocess the data (handle missing values, normalization, etc.).
3. Split the data into training and testing sets using `train_test_split`.
4. Train a logistic regression model on the training data.
5. Evaluate the model on the test set.
6. Display the accuracy and other performance metrics.

## Example Code
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('heart_disease_data.csv')

# Split data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

## Results
- The model's performance is evaluated using accuracy, precision, recall, and F1-score.
- Predictions can help in early detection of heart disease, aiding in medical interventions.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is open-source and available under the MIT License.


