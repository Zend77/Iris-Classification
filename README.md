Iris Classifier Comparison
This project compares the performance of multiple machine learning models on the classic Iris flower classification problem. It includes model training, evaluation, and visualizations to explore decision-making and performance.

Project Structure
├── data_loader.py         # Loads and splits the Iris dataset
├── models.py              # Defines and builds multiple ML pipelines
├── train.py               # Contains training and evaluation functions
├── visualization.py       # Plots data, confusion matrices, and model predictions
├── main.py                # Main script to train and test models
└── plots/                 # Automatically created for saved plots

Features
Load and preprocess the Iris dataset

Train multiple models using Scikit-Learn Pipelines:
Logistic Regression
Support Vector Machine (SVM)
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Evaluate and compare model performance on test data

Visualize:
The raw Iris dataset as a pairplot
Confusion matrix of the best-performing model
Bar chart comparing model accuracy
Scatter plot showing true vs. predicted classifications for each model
Save all visualizations to a /plots directory if interactive display is disabled.

Example Output
Logistic Regression Test acc: 0.97
Support Vector Machine Test acc: 0.97
Decision Tree Test acc: 0.97
Random Forest Test acc: 0.97
K-Nearest Neighbors Test acc: 0.97

Installation
Clone this repository:
git clone https://github.com/your-username/iris-classifier-comparison.git
cd iris-classifier-comparison
Install the required libraries:
pip install -r requirements.txt

Usage
Run the main script with visualization enabled:
python main.py
Run the main script and save plots instead of showing them:
main(show_data_viz=False)
