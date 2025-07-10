from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def build_models():
    
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=200))
        ]),

        'Support Vector Machine': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC())
        ]),

        'Decision Tree': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier())
        ]),

        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ]),

        'K-Nearest Neighbors': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
    }
    return models