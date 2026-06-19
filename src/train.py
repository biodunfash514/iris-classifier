import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

def main():
    #  Setting  up the Argument Parser
    parser = argparse.ArgumentParser(description="Train a Decision Tree on the Iris Dataset")
    
    parser.add_argument("--test-size", type=float, default=0.2 )      
    
    
    parser.add_argument("--random-state", type=int, default=42)
        
        
    args = parser.parse_args()

    # Loading Dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    print(f"Dataset Shape: {df.shape}")
    
    # Visualizing the dataframe 
    
    plt.scatter(df["sepal length (cm)"], df["sepal width (cm)"], c=df["target"])
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title("Iris Dataset Visualization")
    plt.show()

    #  Dataset Preprocessing
    df = df.dropna()
    X = df.drop("target", axis=1)
    y = df["target"]

    # splitting the data into training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    #  Trainin the model
    # Use arguments here for the model as well
    model = DecisionTreeClassifier(random_state=args.random_state)
    model.fit(X_train, y_train)

    #  Evaluation of resultset
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- Model Results ---")      
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save model
    model_path = os.path.join(output_dir, "iris_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Generating and saving the Confusion Matrix Plot 
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred, display_labels=iris.target_names, cmap=plt.cm.Blues, ax=ax)
    
    ax.set_title(f"Confusion Matrix: (test_size={args.test_size})")
    
    fig_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(fig_path)
    plt.close() # Close plot to free up memory
    print(f"figure saved to: {fig_path}")

if __name__ == "__main__":
    main()