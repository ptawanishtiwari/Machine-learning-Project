import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


# TRAINING DATA
trainData = {
    'Hours_study': [2,3,4,5,6,7,8,9,10],
    'Exam_score': [50,60,65,70,75,80,85,90,95]
}

# Create DataFrame
df = pd.DataFrame(trainData)

# Independent & Dependent variables
x = df[['Hours_study']]     
y = df['Exam_score']         
# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create & train model
model = LinearRegression()
model.fit(x_train, y_train)

# User input
user_input = float(input("Enter the Study Hours: "))

# Prediction (2D array required)
prediction = model.predict(np.array([[user_input]]))

print(f"Predicted score is: {prediction[0]:.2f}")
