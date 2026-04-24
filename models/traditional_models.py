import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

class TraditionalModels:
    """Wrapper class for traditional machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.model_instances = {}
        self.training_results = {}
        
    def train_random_forest(self, X_train, y_train, X_test, y_test, n_estimators=100):
        """Train Random Forest model"""
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        self.models['Random Forest'] = rf
        self.training_results['Random Forest'] = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        return rf, accuracy, report
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test, max_iter=1000):
        """Train Logistic Regression model"""
        print("Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=max_iter, n_jobs=-1)
        lr.fit(X_train, y_train)
        
        # Predictions
        y_pred = lr.predict(X_test)
        y_pred_proba = lr.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        self.models['Logistic Regression'] = lr
        self.training_results['Logistic Regression'] = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"Logistic Regression Accuracy: {accuracy:.4f}")
        return lr, accuracy, report
    
    def train_knn(self, X_train, y_train, X_test, y_test, n_neighbors=5):
        """Train K-Nearest Neighbors model"""
        print("Training KNN...")
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        knn.fit(X_train, y_train)
        
        # Predictions
        y_pred = knn.predict(X_test)
        y_pred_proba = knn.predict_proba(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        self.models['KNN'] = knn
        self.training_results['KNN'] = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"KNN Accuracy: {accuracy:.4f}")
        return knn, accuracy, report
    
    def predict(self, model_name, X):
        """Make predictions using trained model"""
        if model_name in self.models:
            return self.models[model_name].predict(X)
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def predict_proba(self, model_name, X):
        """Get prediction probabilities"""
        if model_name in self.models:
            if hasattr(self.models[model_name], 'predict_proba'):
                return self.models[model_name].predict_proba(X)
            else:
                return None
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def save_model(self, model_name, filepath):
        """Save model to disk"""
        if model_name in self.models:
            with open(filepath, 'wb') as f:
                pickle.dump(self.models[model_name], f)
            return True
        return False
    
    def load_model(self, model_name, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            self.models[model_name] = pickle.load(f)
        return self.models[model_name]