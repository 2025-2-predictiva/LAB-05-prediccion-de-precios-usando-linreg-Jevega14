#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_Type: Tipo de combustible.
# - Selling_type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import os
import json
import gzip
import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error


def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path, index_col=False, compression="zip")
    df_test = pd.read_csv(test_path, index_col=False, compression="zip")

    print("Datos cargados exitosamente")
    return df_train, df_test


def preprocess_data(df, dataset_label):
    df["Age"] = 2021 - df["Year"]
    df = df.drop(columns=["Year", "Car_Name"])

    print(f"Preprocesamiento del dataset {dataset_label} completado")
    return df


def split_features_target(df, target_col):
    X_data = df.drop(columns=[target_col])
    y_data = df[target_col]

    print("Separación de características y variable objetivo completada")
    return X_data, y_data


def pipeline_definition(df):
    all_cols = df.columns.tolist()

    categorical_cols = [
        col
        for col in ["Fuel_Type", "Selling_type", "Transmission"]
        if col in all_cols
    ]
    numeric_cols = [col for col in all_cols if col not in categorical_cols]

    print(f"Columnas categóricas: {categorical_cols}")
    print(f"Columnas numéricas: {numeric_cols}")

    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numerical", MinMaxScaler(), numeric_cols),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("feature_selector", SelectKBest(score_func=f_regression)),
            ("linear_model", LinearRegression()),
        ]
    )

    print("Pipeline definido correctamente")
    return model_pipeline


def hyperparameter_optimization(pipeline, X_train, y_train):
    param_grid = {
        "feature_selector__k": list(range(1, X_train.shape[1] + 1)) + ["all"],
        "linear_model__fit_intercept": [True, False],
        "linear_model__positive": [True, False],
    }

    cv_scheme = KFold(n_splits=10, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_scheme,
        scoring="neg_mean_absolute_error",
        verbose=1,
        n_jobs=-1,
        refit=True,
    )

    grid.fit(X_train, y_train)

    print("Optimización de hiperparámetros finalizada")
    return grid


def save_model(best_model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with gzip.open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    print("Modelo guardado exitosamente")


def calculate_metrics(model, X, y, dataset_name):
    predictions = model.predict(X)

    metrics_dict = {
        "type": "metrics",
        "dataset": dataset_name,
        "r2": r2_score(y, predictions),
        "mse": mean_squared_error(y, predictions),
        "mad": median_absolute_error(y, predictions),
    }

    print(f"Métricas calculadas para dataset: {dataset_name}")
    return metrics_dict


def save_metrics(metrics_list, metrics_path):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        for entry in metrics_list:
            f.write(json.dumps(entry) + "\n")

    print("Métricas guardadas exitosamente")


def main():
    train_df, test_df = load_data(
        "files/input/train_data.csv.zip",
        "files/input/test_data.csv.zip",
    )

    train_df = preprocess_data(train_df, "train")
    test_df = preprocess_data(test_df, "test")

    X_train, y_train = split_features_target(train_df, "Present_Price")
    X_test, y_test = split_features_target(test_df, "Present_Price")

    pipeline = pipeline_definition(X_train)
    best_model = hyperparameter_optimization(pipeline, X_train, y_train)

    save_model(best_model, "files/models/model.pkl.gz")

    metrics_train = calculate_metrics(best_model, X_train, y_train, "train")
    metrics_test = calculate_metrics(best_model, X_test, y_test, "test")

    save_metrics([metrics_train, metrics_test], "files/output/metrics.json")


if __name__ == "__main__":
    main()
