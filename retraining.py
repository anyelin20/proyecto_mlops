import numpy as np
import pandas as pd
import os
import json
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime


class PhishingModelOptimizado:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.version = "1.0"
        self.training_date = None
        self.metrics = {}
        self.feature_importances_ = None

    def optimize_hyperparameters(self, X_train, y_train):
        """Optimización de hiperparámetros con GridSearchCV"""
        print("Optimizando hiperparámetros...")

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='roc_auc',
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.feature_importances_ = self.model.feature_importances_

        print("Optimización completada!")
        print(f"Mejores parámetros encontrados: {self.best_params}")

        return self.model

    def train(self, X_train, y_train, optimize=True):
        """Entrenamiento del modelo"""
        if optimize:
            self.optimize_hyperparameters(X_train, y_train)
        else:
            print("Entrenando modelo con parámetros por defecto...")
            self.model = RandomForestClassifier(random_state=42)
            self.model.fit(X_train, y_train)
            self.feature_importances_ = self.model.feature_importances_

        self.training_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("Modelo entrenado exitosamente!")

    def evaluate(self, X_test, y_test):
        """Evaluación del modelo y generación de gráficos"""
        y_pred = self.model.predict(X_test)
        y_probs = self.model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_probs)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.metrics = {
            'roc_auc': roc_auc,
            'classification_report': report,
            'version': self.version,
            'training_date': self.training_date,
            'best_params': self.best_params if self.best_params else "Default"
        }

        self._generate_plots(X_test, y_test, y_probs)

        print(f"\nMétricas de evaluación (v{self.version}):")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        if self.best_params:
            print("Mejores parámetros utilizados:")
            print(self.best_params)

        return self.metrics

    def _generate_plots(self, X_test, y_test, y_probs):
        """Genera gráficos de evaluación"""
        plot_dir = os.path.join("../../modelo_serializado", "plots")
        os.makedirs(plot_dir, exist_ok=True)

        fpr, tpr, _ = roc_curve(y_test, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2,
                 label=f'ROC Curve (AUC = {self.metrics["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - v{self.version}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(plot_dir, f"roc_curve_v{self.version}.png"), bbox_inches='tight', dpi=300)
        plt.close()

        if self.feature_importances_ is not None:
            if isinstance(X_test, pd.DataFrame):
                feature_names = X_test.columns
            else:
                feature_names = [f'feature_{i}' for i in range(len(self.feature_importances_))]

            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.feature_importances_
            }).sort_values('Importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
            plt.title(f'Top 15 Características Importantes - v{self.version}')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"feature_importance_v{self.version}.png"), bbox_inches='tight', dpi=300)
            plt.close()

            self.metrics['top_features'] = feature_importance.head(10).to_dict()

    def save_model(self, directory=r"C:\Users\User\Desktop\MLOPS\proyecto_mlops\modelo_serializado"):
        """Serializa el modelo, métricas y guarda gráficos"""
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(directory, f"phishing_model_opt_v{self.version}_{timestamp}.pkl")
        metrics_filename = os.path.join(directory, f"model_metrics_opt_v{self.version}_{timestamp}.json")

        with open(model_filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'version': self.version,
                'best_params': self.best_params,
                'feature_importances': self.feature_importances_
            }, f)

        with open(metrics_filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        print(f"\nModelo guardado en: {model_filename}")
        print(f"Métricas guardadas en: {metrics_filename}")
        print(f"Gráficos guardados en: {os.path.join(directory, 'plots')}")

        return model_filename


def run_pipeline():
    file_path = r"/proyecto_mlops/data/email_phishing_data.csv"
    data = pd.read_csv(file_path)

    X = data.drop('label', axis=1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    phishing_model = PhishingModelOptimizado()
    phishing_model.train(X_train, y_train, optimize=True)
    phishing_model.evaluate(X_test, y_test)
    phishing_model.save_model()


if __name__ == "__main__":
    run_pipeline()
