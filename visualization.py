import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris

def save_plot(filename):
    os.makedirs("plots", exist_ok=True)
    plt.savefig(os.path.join("plots", filename))
    plt.close()

def visualize_data(show_plot=True):
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = pd.Categorical.from_codes(data.target, data.target_names)

    sns.pairplot(df, hue='species', palette='bright')
    plt.suptitle('Iris Dataset Pair Plot', y=1.02)

    if show_plot:
        plt.show()
    else:
        save_plot("iris_pairplot.png")

def plot_confusion_matrix(y_true, y_pred, show_plot=True):
    data = load_iris()
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=data.target_names, 
                yticklabels=data.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if show_plot:
        plt.show()
    else:
        save_plot("confusion_matrix.png")

def plot_model_performance(results, show_plot=True):
    names = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(names, scores)
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    plt.xticks(rotation=30)

    if show_plot:
        plt.show()
    else:
        save_plot("model_performance.png")
        
def plot_predictions(X_test, y_true, y_pred, model_name, show_plot=True):
    data = load_iris()
    feature_names = data.feature_names

    df = pd.DataFrame(X_test, columns=feature_names)
    df['True Species'] = pd.Categorical.from_codes(y_true, data.target_names)
    df['Predicted Species'] = pd.Categorical.from_codes(y_pred, data.target_names)

    # Plot each feature against the true and predicted labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=feature_names[0], y=feature_names[1], hue='True Species', style='Predicted Species', palette='bright', s=100)
    plt.title(f'{model_name} Predictions vs True Labels')

    if show_plot:
        plt.show()
    else:
        save_plot(f"{model_name.replace(' ', '_').lower()}_predictions.png")
