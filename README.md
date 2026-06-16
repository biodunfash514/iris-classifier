# Iris Species Classifier

The  project trains a Decision Tree model to classify Iris flowers using the Scikit-Learn Iris dataset.

## Installation
1. Create a virtual environment:
   ```bash
   python -m venv env


2. Activate the environment:
   Windows: .\env\Scripts\activate
   Mac/Linux: source env/bin/activate

3. Install dependencies:
   bash
   pip install -r requirements.txt

4. Usage
   Run the training script with custom parameters:

    bash
    python src/train.py --test-size 0.2 --random-state 42

5. Outputs
   Confusion Matrix: Saved to outputs/confusion_matrix.png

6. Testing
  Run the accuracy tests using pytest:

  bash
  pytest accuracy_test.py