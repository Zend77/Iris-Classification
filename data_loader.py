from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, random_state=1):
    data = load_iris()
    X = data.data # type: ignore
    y = data.target # type: ignore
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test