import sys
import os
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from source.exception import CustomException


# ------------------------------------------------------------
# Save object using dill
# ------------------------------------------------------------
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# ------------------------------------------------------------
# Load saved object
# ------------------------------------------------------------
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


# ------------------------------------------------------------
# Evaluate multiple models with GridSearchCV
# ------------------------------------------------------------
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Trains multiple models with hyperparameter tuning and returns their R² test scores.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging_msg = f"Training model: {model_name}"
            print(logging_msg)

            # Fetch corresponding parameters for the model
            para = param.get(model_name, {})

            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)

            # Set best parameters & retrain
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Save only the test R² in report
            report[model_name] = test_model_score

            print(f"{model_name} -> Train R²: {train_model_score:.3f}, Test R²: {test_model_score:.3f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
