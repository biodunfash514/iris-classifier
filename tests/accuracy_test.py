import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def test_model_accuracy_above_90():
    # Data setup
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split (
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Assertion
    assert accuracy > 0.90, f"Model accuracy is too low: {accuracy:.4f}"