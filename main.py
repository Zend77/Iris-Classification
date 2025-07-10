from data_loader import load_data
from models import build_models 
from train import train_model, evaluate_model
from visualization import visualize_data, plot_confusion_matrix, plot_model_performance, plot_predictions

def main(show_data_viz=True):
    
    visualize_data(show_plot=show_data_viz)

    X_train, X_test, y_train, y_test = load_data()

    models = build_models()
    results = {}

    for name, model in models.items():
        train_model(model, X_train, y_train)
        accuracy, y_pred = evaluate_model(model, X_test, y_test)
        print(f"{name} Test acc: {accuracy:.10f}")
        results[name] = accuracy
        
        plot_predictions(X_test, y_test, y_pred, model_name=name, show_plot=show_data_viz)

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    _, best_pred = evaluate_model(best_model, X_test, y_test)
    print(f"Showing confusion matrix for best model: {best_model_name}")
    plot_confusion_matrix(y_test, best_pred, show_plot=show_data_viz)

    plot_model_performance(results, show_plot=show_data_viz)

if __name__ == "__main__":
    main(show_data_viz=False)