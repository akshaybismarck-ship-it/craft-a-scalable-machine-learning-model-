#!/bin/bash

# Import necessary libraries
pip install -q scikit-learn pandas numpy

# Load iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a scalable machine learning model simulator
def scalable_model_simulator(data, target, model, scale_factor):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Scale data using StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Initialize model with scale factor
    model = RandomForestClassifier(n_estimators=scale_factor)
    
    # Train the model
    model.fit(data_scaled, target)
    
    return model

# Test the simulator with varying scale factors
scale_factors = [10, 50, 100, 200]
for scale_factor in scale_factors:
    model = scalable_model_simulator(X_train, y_train, "Random Forest", scale_factor)
    print(f"Scale Factor: {scale_factor}, Accuracy: {model.score(X_test, y_test):.3f}")