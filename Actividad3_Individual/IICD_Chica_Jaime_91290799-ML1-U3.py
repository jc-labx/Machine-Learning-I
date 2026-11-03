#! python

#
# Machine Learning I - Actividad individual - Unidad 3
#
# Autor: Jaime Alberto Chica Betancourt
#
# Universidad de Investigación y Desarrollo – UDI
# Maestría en Ciencia de Datos e Inteligencia Artificial
# 7614 – Machine Learning I
# Docente: Jonathan Arley Monsalve Salazar
#
# Fecha de Entrega: 7 de Noviembre de 2026


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


# Proporción de observaciones utilizadas para prueba
TEST_SIZE = 0.20

# Aleatoriedad para obtener una partición reproducible, de acuerdo a la documentación de scikit-learn
RANDOM_STATE = 42

# Dataset Mall customers
MALL_FILE = "Mall_Customers.csv"

# Dataset Air Passangers
AIR_FILE = "AirPassengers.csv"


# -----------------------------------------------------------------------------
# REGRESIÓN POLINOMIAL
# -----------------------------------------------------------------------------

def create_nonlinear_dataset():

    """
    Genera datos sintéticos con una relación cúbica y ruido aleatorio.
    """

    # Crea un generador aleatorio reproducible
    rng = np.random.default_rng(RANDOM_STATE)

    # Genera 200 valores de X distribuidos entre -3 y 3
    X = rng.uniform(-3, 3, size=200).reshape(-1, 1)

    # Genera ruido aleatorio para simular variabilidad en los datos
    noise = rng.normal(0, 3, size=200)

    # Genera y mediante una relación cúbica con X y añade el ruido
    y = (
        4
        + 2 * X[:, 0]
        - 1.5 * X[:, 0] ** 2
        + 0.8 * X[:, 0] ** 3
        + noise
    )

    return X, y


def train_regression_models(X_train, y_train):

    """
    Entrena modelos lineal y polinomial de grados 2 y 3
    """

    # Define los tres modelos que serán comparados: regresión lineal y regresiones polinomiales de grados 2 y 3
    models = {
        "Lineal": LinearRegression(),
        
        # Genera X y X² antes de aplicar la regresión lineal        
        "Polinomial grado 2": Pipeline([
            ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
            ("linear_regression", LinearRegression()),
        ]),
        
        # Genera X, X² y X³ antes de aplicar la regresión lineal
        "Polinomial grado 3": Pipeline([
            ("polynomial_features", PolynomialFeatures(degree=3, include_bias=False)),
            ("linear_regression", LinearRegression()),
        ]),
    }

    # Entrena los tres modelos utilizando los mismos datos
    for model in models.values():
        model.fit(X_train, y_train)

    return models


def evaluate_regression_models(models, X_train, X_test, y_train, y_test):

    """
    Calcula MSE y R2 de entrenamiento y prueba para cada modelo
    """

    rows = []

    # Genera predicciones y calcula las métricas de entrenamiento y prueba para cada modelo
    for model_name, model in models.items():
        train_prediction = model.predict(X_train)
        test_prediction = model.predict(X_test)

        rows.append({
            "Modelo": model_name,
            "MSE entrenamiento": mean_squared_error(y_train, train_prediction),
            "MSE prueba": mean_squared_error(y_test, test_prediction),
            "R2 entrenamiento": r2_score(y_train, train_prediction),
            "R2 prueba": r2_score(y_test, test_prediction),
        })

    # Convierte los resultados en una tabla para facilitar la comparación
    results = pd.DataFrame(rows)
    print("\nREGRESIÓN: COMPARACIÓN DE MODELOS")
    print("=" * 90)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return results


def plot_regression_fits(X, y, models):

    """
    Grafica los datos y las curvas ajustadas por los tres modelos
    """

    # Genera 500 valores ordenados de X para dibujar de forma continua las curvas de los modelos
    X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, alpha=0.45, label="Datos con ruido")

    # Genera y grafica las predicciones de cada modelo sobre los mismos valores de X
    for model_name, model in models.items():
        plt.plot(X_plot[:, 0], model.predict(X_plot), linewidth=2, label=model_name)

    plt.title("Comparación del ajuste lineal y polinomial")
    plt.xlabel("Variable predictora X")
    plt.ylabel("Variable objetivo y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("regression_fits.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_polynomial_regression():

    """
    Ejecuta el análisis completo de regresión polinomial
    """

    # Genera el conjunto de datos sintético con relación no lineal
    X, y = create_nonlinear_dataset()
    
    # Divide los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,)

    # Entrena los modelos lineal y polinomiales de grados 2 y 3
    models = train_regression_models(X_train, y_train)

    # Evalúa y compara los modelos mediante MSE y R²
    results = evaluate_regression_models(models, X_train, X_test, y_train, y_test)
    plot_regression_fits(X, y, models)

    return {"models": models, "results": results}


# -----------------------------------------------------------------------------
# CLUSTERING
# -----------------------------------------------------------------------------

def load_mall_data(file_path=MALL_FILE):

    """
    Carga el dataset Mall Customers desde un archivo CSV local
    """

    data = pd.read_csv(file_path, dtype={"CustomerID": str})
    print("\nMALL CUSTOMERS")
    print("=" * 90)
    print(f"Observaciones: {data.shape[0]}")
    print(f"Variables: {data.shape[1]}")
    print(f"Valores faltantes: {int(data.isna().sum().sum())}")
    return data


def prepare_clustering_data(data):

    """
    Selecciona ingreso y puntuación de gasto y los estandariza
    """

    feature_names = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = data[feature_names]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler


def find_optimal_k(X_scaled):

    """
    Evalúa K-Means con el método del codo y el coeficiente de silueta
    """
    
    # Evalúa soluciones entre 2 y 10 clusters
    k_values = range(2, 11)
    inertia_values = []
    silhouette_values = []

    # Entrena K-Means para cada valor de k y almacena la inercia y el coeficiente de silueta obtenidos
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertia_values.append(model.inertia_)
        silhouette_values.append(silhouette_score(X_scaled, labels))

    # Selecciona el número de clusters con mayor coeficiente de silueta
    best_k = list(k_values)[int(np.argmax(silhouette_values))]

    # Organiza los resultados para facilitar su comparación
    results = pd.DataFrame({"k": list(k_values), "Inercia": inertia_values, "Silueta": silhouette_values,})

    print("\nK-MEANS: EVALUACIÓN DEL NÚMERO DE CLUSTERS")
    print("=" * 90)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nNúmero de clusters seleccionado por silueta: {best_k}")

    # Grafica el método del codo y el coeficiente de silueta
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(list(k_values), inertia_values, marker="o")
    axes[0].set_title("Método del codo")
    axes[0].set_xlabel("Número de clusters (k)")
    axes[0].set_ylabel("Inercia")
    axes[0].grid(alpha=0.3)

    axes[1].plot(list(k_values), silhouette_values, marker="o")
    axes[1].axvline(best_k, linestyle="--", color="red", label=f"Mejor k = {best_k}")
    axes[1].set_title("Coeficiente de silueta")
    axes[1].set_xlabel("Número de clusters (k)")
    axes[1].set_ylabel("Silueta")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("clustering_k_selection.png", dpi=300, bbox_inches="tight")
    plt.close()

    return best_k, results


def fit_clustering_models(X_scaled, best_k):

    """
    Entrena K-Means y DBSCAN sobre las variables estandarizadas
    """

    # Entrena K-Means utilizando el número de clusters seleccionado previamente mediante el coeficiente de silueta
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    # Entrena DBSCAN utilizando valores iniciales de distancia máxima entre vecinos (eps) y número mínimo de observaciones (min_samples)
    dbscan = DBSCAN(eps=0.35, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # Cuenta los clusters identificados por DBSCAN, excluyendo la etiqueta -1 correspondiente al ruido
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    
    # Cuenta las observaciones identificadas como ruido
    n_noise = int(np.sum(dbscan_labels == -1))

    print("\nRESULTADOS DE CLUSTERING")
    print("=" * 90)
    print(f"Clusters K-Means: {len(np.unique(kmeans_labels))}")
    print(f"Clusters DBSCAN: {n_dbscan_clusters}")
    print(f"Observaciones identificadas como ruido por DBSCAN: {n_noise}")

    return {
        "kmeans": kmeans,
        "kmeans_labels": kmeans_labels,
        "dbscan": dbscan,
        "dbscan_labels": dbscan_labels,
        "dbscan_clusters": n_dbscan_clusters,
        "dbscan_noise": n_noise,
    }


def plot_clusters(X, clustering_results):
    
    """
    Visualiza en 2D los clusters obtenidos por K-Means y DBSCAN
    """
    
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(
        x=X.iloc[:, 0],
        y=X.iloc[:, 1],
        hue=clustering_results["kmeans_labels"],
        palette="tab10",
        ax=axes[0],
        legend="full",
    )
    axes[0].set_title("Clusters con K-Means")
    axes[0].set_xlabel("Ingreso anual (k$)")
    axes[0].set_ylabel("Puntuación de gasto")

    sns.scatterplot(
        x=X.iloc[:, 0],
        y=X.iloc[:, 1],
        hue=clustering_results["dbscan_labels"],
        palette="tab10",
        ax=axes[1],
        legend="full",
    )
    axes[1].set_title("Clusters con DBSCAN (-1 = ruido)")
    axes[1].set_xlabel("Ingreso anual (k$)")
    axes[1].set_ylabel("Puntuación de gasto")

    plt.tight_layout()
    plt.savefig("clustering_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_clustering():
    
    """
    Ejecuta el análisis completo de clustering
    """
    
    # Carga el dataset Mall Customers
    data = load_mall_data()

    # Selecciona las variables de ingreso y gasto y las estandariza    
    X, X_scaled, scaler = prepare_clustering_data(data)

    # Evalúa diferentes valores de k y selecciona el mejor mediante el coeficiente de silueta
    best_k, k_results = find_optimal_k(X_scaled)

    # Entrena K-Means con el k seleccionado y DBSCAN sobre los mismos datos estandarizados
    clustering_results = fit_clustering_models(X_scaled, best_k)
    
    # Compara visualmente los clusters obtenidos por ambos métodos    
    plot_clusters(X, clustering_results)

    return {
        "data": data,
        "scaler": scaler,
        "k_results": k_results,
        **clustering_results,
    }


# -----------------------------------------------------------------------------
# SERIES DE TIEMPO
# -----------------------------------------------------------------------------

def load_air_passengers(file_path=AIR_FILE):

    """
    Carga AirPassengers y configura la fecha como índice mensual
    """
    
    data = pd.read_csv(file_path)
    data["Month"] = pd.to_datetime(data["Month"])
    data = data.set_index("Month").asfreq("MS")
    series = data["#Passengers"].astype(float)

    print("\nAIR PASSENGERS")
    print("=" * 90)
    print(f"Observaciones mensuales: {len(series)}")
    print(f"Valores faltantes: {int(series.isna().sum())}")
    print(f"Periodo: {series.index.min().date()} a {series.index.max().date()}")
    return series


def plot_time_series(series):
    
    """
    Grafica la serie original y sus medias móviles de 6 y 12 meses
    """
    
    plt.figure(figsize=(12, 6))
    plt.plot(series, label="Pasajeros mensuales", linewidth=1.8)
    plt.plot(series.rolling(6).mean(), label="Media móvil 6 meses")
    plt.plot(series.rolling(12).mean(), label="Media móvil 12 meses")
    plt.title("AirPassengers: serie y medias móviles")
    plt.xlabel("Fecha")
    plt.ylabel("Número de pasajeros")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("time_series_moving_average.png", dpi=300, bbox_inches="tight")
    plt.close()


def analyze_time_series(series):
    
    """
    Genera autocorrelación, descomposición y prueba ADF
    """
    
    figure, axis = plt.subplots(figsize=(10, 5))
    plot_acf(series, lags=36, ax=axis)
    axis.set_title("Autocorrelación de AirPassengers")
    axis.set_xlabel("Retardo (meses)")
    axis.set_ylabel("Autocorrelación")
    plt.tight_layout()
    plt.savefig("time_series_autocorrelation.png", dpi=300, bbox_inches="tight")
    plt.close()


    # Descompone la serie en tendencia, estacionalidad y residuos, utilizando un ciclo estacional de 12 meses
    decomposition = seasonal_decompose(series, model="multiplicative", period=12)

    decomposition_figure = decomposition.plot()
    decomposition_figure.set_size_inches(12, 9)
    decomposition_figure.tight_layout()
    decomposition_figure.savefig("time_series_decomposition.png", dpi=300, bbox_inches="tight",)
    plt.close(decomposition_figure)

    # Aplica la prueba ADF para evaluar si la serie es estacionaria
    adf_statistic, p_value, _, _, critical_values, _ = adfuller(series, result_object=False)

    print("\nPRUEBA DE DICKEY-FULLER AUMENTADA")
    print("=" * 90)
    print(f"Estadístico ADF: {adf_statistic:.4f}")
    print(f"Valor p: {p_value:.4f}")
    for level, value in critical_values.items():
        print(f"Valor crítico {level}: {value:.4f}")

    # Con alfa = 0,05, un valor p menor a 0,05 permite considerar la serie estacionaria
    stationarity = "estacionaria" if p_value < 0.05 else "no estacionaria"
    print(f"Conclusión con alfa=0.05: la serie es {stationarity}.")

    return {
        "decomposition": decomposition,
        "adf_statistic": adf_statistic,
        "p_value": p_value,
        "stationarity": stationarity,
    }


def forecast_time_series(series, test_months=24):

    """
    Pronostica los últimos meses mediante Holt-Winters con tendencia aditiva y estacionalidad multiplicativa
    """

    # Utiliza los últimos 24 meses como conjunto de prueba y todos los meses anteriores como entrenamiento
    train = series.iloc[:-test_months]
    test = series.iloc[-test_months:]

    # Entrena Holt-Winters considerando tendencia creciente y un patrón estacional que se repite cada 12 meses
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="mul",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True)

    # Pronostica los meses reservados para prueba
    forecast = model.forecast(test_months)

    # Compara el pronóstico con los valores reales
    mse = mean_squared_error(test, forecast)
    r2 = r2_score(test, forecast)

    print("\nPRONÓSTICO HOLT-WINTERS")
    print("=" * 90)
    print(f"Meses pronosticados: {test_months}")
    print(f"MSE de prueba: {mse:.4f}")
    print(f"R2 de prueba: {r2:.4f}")

    # Compara visualmente entrenamiento, valores reales y pronóstico del modelo
    plt.figure(figsize=(12, 6))
    plt.plot(train, label="Entrenamiento")
    plt.plot(test, label="Valores reales", linewidth=2)
    plt.plot(forecast, label="Pronóstico Holt-Winters", linestyle="--", linewidth=2)
    plt.title("Pronóstico de pasajeros aéreos")
    plt.xlabel("Fecha")
    plt.ylabel("Número de pasajeros")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("time_series_forecast.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "model": model,
        "train": train,
        "test": test,
        "forecast": forecast,
        "mse": mse,
        "r2": r2,
    }


def run_time_series():

    """
    Ejecuta el análisis completo de la serie temporal
    """

    # Carga la serie temporal AirPassengers
    series = load_air_passengers()
    
    # Grafica la serie original y sus medias móviles
    plot_time_series(series)
    
    # Analiza autocorrelación, tendencia, estacionalidad y estacionariedad de la serie
    analysis = analyze_time_series(series)

    # Entrena Holt-Winters y evalúa el pronóstico utilizando los últimos 24 meses como prueba
    forecast = forecast_time_series(series)

    return {"series": series, "analysis": analysis, "forecast": forecast}


if __name__ == "__main__":

    # 1. Ejecuta el análisis de regresión lineal y polinomial
    regression_results = run_polynomial_regression()

    # 2. Ejecuta el análisis de clustering con K-Means y DBSCAN
    clustering_results = run_clustering()

    # 3. Ejecuta el análisis y pronóstico de la serie temporal
    time_series_results = run_time_series()

    print("\nAnálisis completado.")


