#! python

#
# Machine Learning I - Actividad individual - Unidad 1
#
# Autor: Jaime Alberto Chica Betancourt
#
# Universidad de Investigación y Desarrollo – UDI
# Maestría en Ciencia de Datos e Inteligencia Artificial
# 7614 – Machine Learning I
# Docente: Jonathan Arley Monsalve Salazar
#
# Fecha de Entrega: 18 de Octubre de 2026
#

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Variable objetivo (la que queremos predecir)
VAL_TARGET = "MedHouseVal"

# Predictor seleccionado para la regresión lineal simple: El EDA mostró que era la variable con mayor correlación positiva con MedHouseVal
VAL_PREDICTOR = "MedInc"

# % de observaciones del modelo a ser utilizadas para entrenamiento
TEST_SIZE = 0.20

# Aleatoriedad para obtener una partición reproducible, de acuerdo a la documentación de scikit-learn
RANDOM_STATE = 42

# Valores de regularización que serán evaluados para Ridge y Lasso
ALPHAS = [
    0.001,
    0.01,
    0.1,
    1,
    10,
    100,
    1000
]

def load_data():
    """
    Carga el California Housing Dataset y devuelve un DataFrame de Pandas.
    """
    # Carga el dataset como DataFrame de Pandas
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    return df


def explore_data(df):
    """
    Realiza una inspección preliminar del conjunto de datos.
    """
    print("\nCALIFORNIA HOUSING DATASET")
    print("=" * 70)

    print("\nDimensiones del dataset:")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    print("\nPrimeras 5 filas:")
    print(df.head())

    print("\nInformación general:")
    df.info()

    print("\nEstadísticas descriptivas:")
    print(df.describe().T)

    print("\nValores nulos por variable:")
    print(df.isnull().sum())


def plot_histograms(df):
    """
    Genera histogramas para todas las variables del dataset.
    """
    print("\nGenerando histogramas...")
    df.hist(bins=30, figsize=(15, 10), edgecolor="black")
    plt.suptitle("Distribución de las variables - California Housing", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("histograms.png", dpi=300, bbox_inches="tight")


def plot_boxplots(df):
    """
    Genera boxplots individuales para detectar posibles valores extremos en las variables.
    """
    print("Generando boxplots...")

    # Crea una cuadrícula de 3x3 gráficos
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, column in enumerate(df.columns):
        sns.boxplot(y=df[column], ax=axes[i])
        axes[i].set_title(column)
        axes[i].set_ylabel("")

    # Oculta posibles subplots sobrantes
    for i in range(len(df.columns), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Boxplots de las variables - California Housing", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("boxplots.png", dpi=300, bbox_inches="tight")


def plot_correlation_matrix(df):
    """
    Genera la matriz de correlación entre todas las variables e imprime las correlaciones con la variable objetivo.
    """
    print("Generando matriz de correlación...")

    correlation = df.corr()

    plt.figure(figsize=(11, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Matriz de correlación - California Housing", fontsize=16)
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=300, bbox_inches="tight")

    # Correlaciones respecto a la variable objetivo
    target_correlation = (
        correlation[VAL_TARGET]
        .drop(VAL_TARGET)
        .sort_values(ascending=False)
    )

    print("\nCorrelación de las variables con MedHouseVal:")
    print(target_correlation)

    return target_correlation


def plot_target_relationships(df):
    """
    Genera scatterplots entre cada predictor y MedHouseVal.
    """
    print("\nGenerando relaciones con la variable objetivo...")

    # Selecciona todas las variables predictoras, excluyendo la variable objetivo
    predictors = [
        column
        for column in df.columns
        if column != VAL_TARGET
    ]

    # Crea una cuadrícula de 2x4 gráficos
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    # Genera un scatterplot de cada predictor frente a MedHouseVal
    for i, column in enumerate(predictors):
        sns.scatterplot(data=df, x=column, y=VAL_TARGET, alpha=0.3, s=15, ax=axes[i])
        axes[i].set_title(f"{column} vs {VAL_TARGET}")

    plt.suptitle("Relación entre predictores y MedHouseVal", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("target_relationships.png", dpi=300, bbox_inches="tight")


def plot_geographical_distribution(df):
    """
    Visualiza la distribución geográfica de los registros del dataset.
    Longitude se utiliza como eje X y Latitude como eje Y.
    El color representa el valor de MedHouseVal.
    """
    print("Generando distribución geográfica...")

    plt.figure(figsize=(10, 8))

    # Grafica geográficamente las observaciones
    scatter = plt.scatter(df["Longitude"], df["Latitude"], c=df[VAL_TARGET], cmap="viridis", alpha=0.5, s=10)

    plt.colorbar(scatter, label="MedHouseVal")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Distribución geográfica del valor medio de vivienda", fontsize=15)
    plt.tight_layout()
    plt.savefig("geographical_distribution.png", dpi=300, bbox_inches="tight")


def evaluate_model(
    model_name,
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    alpha=None
):
    """
    Calcula MSE y R² para los datos de entrenamiento y prueba del modelo seleccionado
    """
    metrics = {
        "Model": model_name,
        "Alpha": alpha,
        "MSE Train": mean_squared_error(y_train, y_train_pred),
        "MSE Test": mean_squared_error(y_test, y_test_pred),
        "R2 Train": r2_score(y_train, y_train_pred),
        "R2 Test": r2_score(y_test, y_test_pred)
    }

    print(f"\nResultados - {model_name}")
    print("-" * 50)

    if alpha is not None:
        print(f"Alpha     : {alpha}")

    print(f"MSE Train : {metrics['MSE Train']:.4f}")
    print(f"MSE Test  : {metrics['MSE Test']:.4f}")
    print(f"R² Train  : {metrics['R2 Train']:.4f}")
    print(f"R² Test   : {metrics['R2 Test']:.4f}")

    return metrics


def simple_linear_regression(df):
    """
    Entrena una regresión lineal simple usando VAL_PREDICTOR para predecir VAL_TARGET.
    """
    print("\nREGRESIÓN LINEAL SIMPLE")
    print("=" * 70)

    # Separa la variable predictora (X) y la variable objetivo (y)
    X = df[[VAL_PREDICTOR]]
    y = df[VAL_TARGET]

    # Divide los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Crea el modelo y estima intercepto y coeficiente usando solamente los datos de entrenamiento.  La predicción se estima de acuerdo a y_hat​ = b0 ​+ b1*​x
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Genera predicciones sobre entrenamiento y prueba
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"\nVariable predictora: {VAL_PREDICTOR}")
    print(f"Coeficiente: {model.coef_[0]:.4f}")
    print(f"Intercepto : {model.intercept_:.4f}")

    metrics = evaluate_model("Regresión lineal simple", y_train, y_train_pred, y_test, y_test_pred)

    # Genera 200 valores equidistantes del predictor para dibujar la línea estimada por la regresión
    x_line = np.linspace(X[VAL_PREDICTOR].min(), X[VAL_PREDICTOR].max(), 200)

    # Calcula la predicción del modelo para cada punto de la línea
    x_line_df = pd.DataFrame({VAL_PREDICTOR: x_line})
    y_line = model.predict(x_line_df)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[VAL_PREDICTOR], y_test, alpha=0.3, s=15, label="Valores reales")
    plt.plot(x_line, y_line, color="red", linewidth=2, label="Línea de regresión")
    plt.xlabel(VAL_PREDICTOR)
    plt.ylabel(VAL_TARGET)
    plt.title(f"Regresión lineal simple: {VAL_PREDICTOR} vs {VAL_TARGET}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("simple_linear_regression.png", dpi=300, bbox_inches="tight")

    return model, metrics


def multiple_linear_regression(df):
    """
    Entrena una regresión lineal múltiple utilizando todas las variables predictoras excepto VAL_TARGET
    """

    print("\nREGRESIÓN LINEAL MÚLTIPLE")
    print("=" * 70)

    # Separa las variables predictoras (X) de la variable objetivo (y)
    X = df.drop(columns=[VAL_TARGET])
    y = df[VAL_TARGET]

    # Divide los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Crea el modelo y estima un coeficiente para cada predictor usando solamente los datos de entrenamiento.  La predicción se estima de acuerdo a y_hat = b0 + b1*X1 + b2*X2 + ... + b8*X8​
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Genera predicciones sobre entrenamiento y prueba
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"\nIntercepto: {model.intercept_:.4f}")

    # Compara los valores reales con las predicciones del modelo
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.3, s=15)

    # Línea de referencia que representa una predicción perfecta
    min_value = min(y_test.min(), y_test_pred.min())
    max_value = max(y_test.max(), y_test_pred.max())

    plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--", linewidth=2, label="Predicción perfecta")
    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.title("Regresión lineal múltiple: valores reales vs. predichos")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("multiple_linear_regression.png", dpi=300, bbox_inches="tight")

    # Asocia cada variable predictora con el coeficiente estimado por el modelo
    coefficients = pd.DataFrame({"Variable": X.columns, "Coeficiente": model.coef_})

    # Ordena los coeficientes por magnitud para facilitar su inspección.
    # Al estar en escalas distintas, esta magnitud no representa directamente la importancia relativa de las variables.
    coefficients["Valor absoluto"] = coefficients["Coeficiente"].abs()
    coefficients = coefficients.sort_values(by="Valor absoluto", ascending=False)

    print("\nCoeficientes del modelo:")
    print(coefficients[["Variable", "Coeficiente"]].to_string(index=False))

    metrics = evaluate_model(
        "Regresión lineal múltiple",
        y_train,
        y_train_pred,
        y_test,
        y_test_pred
    )

    return model, metrics, coefficients


def find_best_regularized_model(
    model_name,
    model,
    X_train,
    y_train
):

    """
    Busca el mejor alpha mediante validación cruzada.
    El escalado se incluye dentro del Pipeline para evitar que los datos de validación lo influyan
    """

    # Encadena el escalado y el modelo para que ambos formen parte del proceso de validación cruzada
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

    # Define los valores de alpha que se evaluarán
    # "model__alpha" accede al parámetro alpha del modelo dentro del Pipeline
    parameter_grid = {"model__alpha": ALPHAS}

    # Busca el alpha con menor MSE mediante validación cruzada de 5 particiones
    # scikit-learn usa el MSE negativo porque GridSearchCV maximiza el scoring
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1
    )

    # Evalúa cada alpha usando únicamente el conjunto de entrenamiento
    search.fit(X_train, y_train)

    # Recupera el alpha asociado al mejor resultado de validación cruzada
    best_alpha = search.best_params_["model__alpha"]

    print(f"\nMejor alpha para {model_name}: {best_alpha}")

    return search.best_estimator_, best_alpha


def regularized_regressions(df):
    """
    Entrena y evalúa Ridge y Lasso.
    También entrena una regresión lineal sobre variables estandarizadas para comparar correctamente los coeficientes.
    """

    print("\nREGULARIZACIÓN: RIDGE Y LASSO")
    print("=" * 70)

    # Separa las variables predictoras (X) de la variable objetivo (y)
    X = df.drop(columns=[VAL_TARGET])
    y = df[VAL_TARGET]

    # Divide los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Estandariza los predictores antes de aplicar la regresión lineal.
    # Esto permite comparar posteriormente sus coeficientes con Ridge y Lasso bajo una misma escala
    standardized_linear = Pipeline([("scaler", StandardScaler()),("model", LinearRegression())])
    standardized_linear.fit(X_train, y_train)

    # Busca mediante validación cruzada el mejor alpha para Ridge
    # La función devuelve el Pipeline entrenado y el alpha seleccionado
    ridge_model, ridge_alpha = find_best_regularized_model(model_name="Ridge", model=Ridge(), X_train=X_train, y_train=y_train)

    # Evalúa el mejor modelo Ridge sobre entrenamiento y prueba
    ridge_train_pred = ridge_model.predict(X_train)
    ridge_test_pred = ridge_model.predict(X_test)
    ridge_metrics = evaluate_model("Ridge", y_train, ridge_train_pred, y_test, ridge_test_pred, alpha=ridge_alpha)

    # Busca mediante validación cruzada el mejor alpha para Lasso
    lasso_model, lasso_alpha = find_best_regularized_model(
        model_name="Lasso",
        model=Lasso(max_iter=50000),
        X_train=X_train,
        y_train=y_train
    )

    # Evalúa el mejor modelo Lasso sobre entrenamiento y prueba
    lasso_train_pred = lasso_model.predict(X_train)
    lasso_test_pred = lasso_model.predict(X_test)
    lasso_metrics = evaluate_model("Lasso", y_train, lasso_train_pred, y_test, lasso_test_pred, alpha=lasso_alpha)

    # Extrae del Pipeline los coeficientes estimados por cada modelo
    # Al utilizar predictores estandarizados, sus magnitudes son comparables
    linear_coefficients = (
        standardized_linear
        .named_steps["model"]
        .coef_
    )

    ridge_coefficients = (
        ridge_model
        .named_steps["model"]
        .coef_
    )

    lasso_coefficients = (
        lasso_model
        .named_steps["model"]
        .coef_
    )

    # Organiza los coeficientes estandarizados para comparar el efecto de la regularización entre los modelos
    coefficient_comparison = pd.DataFrame({
        "Variable": X.columns,
        "Lineal": linear_coefficients,
        "Ridge": ridge_coefficients,
        "Lasso": lasso_coefficients
    })

    # Cuenta cuántos coeficientes Lasso son cero o numéricamente indistinguibles de cero dentro de una pequeña tolerancia
    print("\nComparación de coeficientes estandarizados:")
    print(coefficient_comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    zero_coefficients = np.sum(np.isclose(lasso_coefficients, 0.0, atol=1e-8))

    print(f"\nCoeficientes reducidos a cero por Lasso: {zero_coefficients}")
    plot_coefficient_comparison(coefficient_comparison)
    plot_regularization_paths(X_train, y_train, X.columns)

    return {
        "ridge_model": ridge_model,
        "lasso_model": lasso_model,
        "ridge_metrics": ridge_metrics,
        "lasso_metrics": lasso_metrics,
        "coefficient_comparison": coefficient_comparison
    }


def plot_coefficient_comparison(coefficient_comparison):
    """
    Compara los coeficientes del modelo lineal estandarizado, Ridge y Lasso.
    """

    print("\nGenerando comparación de coeficientes...")
    
    # Usa los nombres de las variables como categorías del eje X
    plot_data = coefficient_comparison.set_index("Variable")

    # Representa conjuntamente los coeficientes de Lineal, Ridge y Lasso para comparar visualmente el efecto de la regularización
    ax = plot_data.plot(kind="bar", figsize=(13, 7), width=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8)

    plt.title("Comparación de coeficientes estandarizados")
    plt.xlabel("Variable predictora")
    plt.ylabel("Coeficiente")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Modelo")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("coefficient_comparison.png", dpi=300, bbox_inches="tight")


def plot_regularization_paths(
    X_train,
    y_train,
    feature_names
):

    """
    Muestra cómo cambian los coeficientes de Ridge y Lasso cuando aumenta alpha.
    """
    print("Generando trayectorias de regularización...")

    # Estandariza los predictores para que la regularización actúe sobre variables expresadas en una escala comparable
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Genera 50 valores de alpha en escala logarítmica entre 10^-4 y 10^3 para observar el efecto desde una regularización muy débil hasta una muy fuerte
    path_alphas = np.logspace(-4, 3, 50)

    ridge_paths = []
    lasso_paths = []

    # Entrena Ridge y Lasso para cada valor de alpha y almacena sus coeficientes para observar cómo cambia la regularización
    for alpha in path_alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        ridge_paths.append(ridge.coef_)
        lasso = Lasso(alpha=alpha, max_iter=50000)
        lasso.fit(X_train_scaled, y_train)
        lasso_paths.append(lasso.coef_)

    ridge_paths = np.array(ridge_paths)
    lasso_paths = np.array(lasso_paths)

    # Crea una cuadrícula de 1x2 gráficos
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    for index, variable in enumerate(feature_names):
        axes[0].plot(path_alphas, ridge_paths[:, index], label=variable)
        axes[1].plot(path_alphas, lasso_paths[:, index], label=variable)

    # Alpha cubre varios órdenes de magnitud, por lo que se representa sobre un eje X logarítmico
    axes[0].set_xscale("log")
    axes[0].set_title("Efecto de alpha sobre los coeficientes Ridge")
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Coeficiente")
    axes[0].axhline(y=0, color="black", linewidth=0.8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xscale("log")
    axes[1].set_title("Efecto de alpha sobre los coeficientes Lasso")
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("Coeficiente")
    axes[1].axhline(y=0, color="black", linewidth=0.8)
    axes[1].grid(alpha=0.3)

    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("regularization_paths.png", dpi=300, bbox_inches="tight")


def run_regression_models(df):
    """
    Ejecuta todos los modelos de regresión.
    """

    # Ejecuta la regresión lineal simple y recupera el modelo y sus métricas
    simple_model, simple_metrics = (simple_linear_regression(df))
    
    # Ejecuta la regresión lineal múltiple y recupera sus coeficientes en las escalas originales
    (multiple_model, multiple_metrics, original_coefficients) = multiple_linear_regression(df)

    # Ejecuta Ridge y Lasso, incluyendo estandarización, selección de alpha, evaluación y comparación de coeficientes
    regularized_results = regularized_regressions(df)

    # Extrae las métricas de los modelos regularizados
    ridge_metrics = regularized_results["ridge_metrics"]
    lasso_metrics = regularized_results["lasso_metrics"]

    # Consolida las métricas de todos los modelos
    results = pd.DataFrame([simple_metrics, multiple_metrics, ridge_metrics, lasso_metrics])

    # Selecciona y ordena las columnas que se mostrarán en la comparación
    columns = ["Model", "Alpha", "MSE Train", "MSE Test", "R2 Train", "R2 Test"]

    results = results[columns]

    print("\nCOMPARACIÓN DE MODELOS")
    print("=" * 70)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    return {
        "simple_model": simple_model,
        "multiple_model": multiple_model,
        "ridge_model": regularized_results["ridge_model"],
        "lasso_model": regularized_results["lasso_model"],
        "original_coefficients": original_coefficients,
        "standardized_coefficients": regularized_results["coefficient_comparison"],
        "results": results
    }


if __name__ == "__main__":

    # 1. Carga e inspección de datos
    print("\nCargando dataset...")
    df = load_data()

    explore_data(df)
    print("\nVariables disponibles:")
    print(list(df.columns))

    # 2. Análisis exploratorio de datos
    print("\nANÁLISIS EXPLORATORIO DE DATOS")
    print("=" * 70)

    plot_histograms(df)
    plot_boxplots(df)
    target_correlation = plot_correlation_matrix(df)
    plot_target_relationships(df)
    plot_geographical_distribution(df)
    print("\nGráficas generadas.")

    print(f"\nVariables ordenadas según su correlación con" f"{VAL_TARGET}:")
    print(target_correlation)
    
    # 3. Entrenamiento, regularización y evaluación de los modelos    
    regression_results = run_regression_models(df)
    
    print("\nAnálisis completado.")