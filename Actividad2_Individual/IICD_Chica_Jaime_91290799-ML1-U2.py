#! python

#
# Machine Learning I - Actividad individual - Unidad 2
#
# Autor: Jaime Alberto Chica Betancourt
#
# Universidad de Investigación y Desarrollo – UDI
# Maestría en Ciencia de Datos e Inteligencia Artificial
# 7614 – Machine Learning I
# Docente: Jonathan Arley Monsalve Salazar
#
# Fecha de Entrega: 25 de Octubre de 2026
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# Variable objetivo (la que queremos predecir)
VAL_TARGET = "target"

# Proporción de observaciones utilizadas para prueba
TEST_SIZE = 0.20

# Aleatoriedad para obtener una partición reproducible, de acuerdo a la documentación de scikit-learn
RANDOM_STATE = 42

# Tabla de traducción entre valores numéricos de target y diagnósticos
CLASS_NAMES = ["Maligno", "Benigno"]


def load_data():
    """
    Carga el dataset como DataFrame e incluye la variable objetivo.
    """
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame.copy()
    return df, cancer


def explore_data(df, cancer):
    """
    Muestra estructura, estadísticas y distribución de las clases.
    """
    print("\nDATASET BREAST CANCER WISCONSIN")
    print("=" * 70)
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")
    print(f"Variables predictoras: {df.shape[1] - 1}")
    print(f"Clases: {list(cancer.target_names)}")

    print("\nPrimeras cinco filas:")
    print(df.head())

    print("\nInformación general:")
    df.info()

    print("\nEstadísticas descriptivas:")
    print(df.describe().T)

    print("\nValores nulos por variable:")
    print(df.isnull().sum())

    print("\nDistribución de las clases:")
    class_counts = df[VAL_TARGET].value_counts().sort_index()
    for class_value, count in class_counts.items():
        print(f"{class_value} = {CLASS_NAMES[class_value]}: {count}")


def plot_class_distribution(df):
    """
    Grafica la cantidad de observaciones malignas y benignas.
    """
    counts = df[VAL_TARGET].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=CLASS_NAMES, y=counts.values, hue=CLASS_NAMES, palette=["#C44E52", "#4C72B0"], legend=False)

    for index, value in enumerate(counts.values):
        ax.text(index, value + 5, str(value), ha="center")

    plt.title("Distribución de las clases")
    plt.xlabel("Diagnóstico")
    plt.ylabel("Número de observaciones")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=300, bbox_inches="tight")


def plot_feature_histograms(df):
    """
    Genera histogramas para las 30 variables predictoras.
    """
    predictors = df.drop(columns=[VAL_TARGET])
    predictors.hist(bins=20, figsize=(18, 16), edgecolor="black")
    plt.suptitle("Distribución de las variables predictoras", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("feature_histograms.png", dpi=300, bbox_inches="tight")


def plot_correlation_matrix(df):
    """
    Visualiza las correlaciones entre predictores y con la variable objetivo.
    """
    correlation = df.corr()

    # La matriz completa permite observar grupos de variables relacionadas.
    plt.figure(figsize=(18, 15))
    sns.heatmap(correlation, cmap="coolwarm", center=0, square=False, cbar_kws={"shrink": 0.8})
    plt.title("Matriz de correlación - Breast Cancer Wisconsin", fontsize=16)
    plt.tight_layout()
    plt.savefig("correlation_matrix_rocinante.png", dpi=300, bbox_inches="tight")

    # Ordena las correlaciones con target para identificar las más relevantes.
    target_correlation = (correlation[VAL_TARGET]
        .drop(VAL_TARGET)
        .sort_values(key=np.abs, ascending=False)
    )

    print("\nVariables con mayor correlación absoluta respecto a target:")
    print(target_correlation.head(10))

    plt.figure(figsize=(10, 7))
    top_correlations = target_correlation.head(15).sort_values()
    colors = ["#C44E52" if value < 0 else "#4C72B0" for value in top_correlations]
    top_correlations.plot(kind="barh", color=colors)
    plt.title("Variables con mayor correlación respecto a target")
    plt.xlabel("Coeficiente de correlación")
    plt.ylabel("Variable")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig("target_correlations.png", dpi=300, bbox_inches="tight")

    return target_correlation


def split_data(df):
    """
    Separa predictores y objetivo, conservando la proporción de clases.
    """
    
    # Separa las variables predictoras (X) de la variable objetivo (y)
    X = df.drop(columns=[VAL_TARGET])
    y = df[VAL_TARGET]

    # Divide los datos en entrenamiento y prueba
    # stratify=y mantiene proporciones similares de malignos y benignos en los conjuntos de entrenamiento y prueba
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def print_confusion_matrix_details(matrix):
    """
    Muestra la matriz de confusión y explica el significado
    de cada uno de sus valores.
    """

    malignant_correct = matrix[0, 0]
    malignant_as_benign = matrix[0, 1]
    benign_as_malignant = matrix[1, 0]
    benign_correct = matrix[1, 1]

    print("\n                        PREDICCIÓN")
    print("                    Maligno   Benigno")
    print(
        f"REAL    Maligno       {malignant_correct:3d}"
        f"       {malignant_as_benign:3d}"
    )
    print(
        f"        Benigno       {benign_as_malignant:3d}"
        f"       {benign_correct:3d}"
    )

    print(f"\nMalignos clasificados correctamente como malignos:\t{malignant_correct}")
    print(f"Benignos clasificados correctamente como benignos:\t{benign_correct}")
    print(f"Malignos clasificados incorrectamente como benignos:\t{malignant_as_benign}")
    print(f"Benignos clasificados incorrectamente como malignos:\t{benign_as_malignant}")



def evaluate_classifier(model_name, model, X_train, X_test, y_train, y_test):
    """
    Calcula accuracy, matriz de confusión, probabilidades, ROC y AUC.
    """

    # Genera predicciones sobre entrenamiento y prueba    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcula las probabilidades estimadas para cada clase,
    # y selecciona solamente la columna 1 (Benigno) para la construcción de ROC/AUC
    y_test_probability = model.predict_proba(X_test)[:, 1]

    # Calcula la proporción de clasificaciones correctas en los datos de entrenamiento y prueba
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    # Construye la matriz de confusión, comparando el diagnóstico real con el diagnóstico que predijo el modelo
    matrix = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    
    # Calcula los puntos necesarios para dibujar la curva ROC
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_test_probability, pos_label=1)

    # Calcula el área bajo la curva ROC usando los diagnósticos reales y las probabilidades que estimó el modelo
    auc = roc_auc_score(y_test, y_test_probability)

    print(f"\nResultados - {model_name}")
    print("-" * 50)
    print(f"Accuracy Train: {accuracy_train:.4f}")
    print(f"Accuracy Test : {accuracy_test:.4f}")
    print(f"AUC           : {auc:.4f}")
    print_confusion_matrix_details(matrix)

    return {
        "Model": model_name,
        "Accuracy Train": accuracy_train,
        "Accuracy Test": accuracy_test,
        "AUC": auc,
        "Confusion Matrix": matrix,
        "FPR": false_positive_rate,
        "TPR": true_positive_rate
    }


def train_logistic_regression(X_train, y_train):
    """
    Entrena un clasificador de regresión logística.
    """

    # Crea una secuencia que estandariza los predictores y entrena al clasificador de regresión logística
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ])

    # Entrena el modelo
    model.fit(X_train, y_train)

    return model
    

def train_naive_bayes(X_train, y_train):
    """
    Entrena un clasificador Naive Bayes con distribución gaussiana.
    """

    # Crea un clasificador Naive Bayes con distribución gaussiana
    model = GaussianNB()

    # Entrena el modelo
    model.fit(X_train, y_train)

    return model


def plot_roc_comparison(results):
    """
    Compara las curvas ROC de ambos modelos en una sola figura.
    """
    plt.figure(figsize=(8, 6))

    for result in results:
        plt.plot(result["FPR"], result["TPR"], linewidth=2, label=f'{result["Model"]} (AUC = {result["AUC"]:.4f})')

    # La diagonal representa el comportamiento de un clasificador aleatorio.
    plt.plot([0, 1], [0, 1], "k--", label="Clasificador aleatorio")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Comparación de curvas ROC - clase positiva: Benigno")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_comparison.png", dpi=300, bbox_inches="tight")


def compare_models(results):
    """
    Construye la tabla comparativa e identifica los modelos con mayor accuracy de prueba y mayor AUC.
    """
    comparison = pd.DataFrame([
        {
            "Model": result["Model"],
            "Accuracy Train": result["Accuracy Train"],
            "Accuracy Test": result["Accuracy Test"],
            "AUC": result["AUC"]
        }
        for result in results
    ])

    comparison["Accuracy Gap"] = (comparison["Accuracy Train"] - comparison["Accuracy Test"]).abs()

    print("\nCOMPARACIÓN DE MODELOS")
    print("=" * 70)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    # Busca la mejor precisión obtenida
    best_accuracy = comparison.loc[comparison["Accuracy Test"].idxmax(), "Model"]
    # Busca la mejor AUC obtenida
    best_auc = comparison.loc[comparison["AUC"].idxmax(), "Model"]

    print(f"\nMayor accuracy de prueba: {best_accuracy}")
    print(f"Mayor AUC: {best_auc}")

    return comparison


def run_models(df):
    """
    Entrena, evalúa y compara Logistic Regression y GaussianNB.
    """

    # Divide los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(df)

    print("\nDISTRIBUCIÓN DESPUÉS DE LA PARTICIÓN")
    print("=" * 70)
    print("Entrenamiento:")
    print(y_train.value_counts(normalize=True).sort_index())
    print("Prueba:")
    print(y_test.value_counts(normalize=True).sort_index())

    # Entrena el modelo de regresión logística
    logistic_model = train_logistic_regression(X_train, y_train)
    # Evalúa los resultados
    logistic_results = evaluate_classifier("Regresión logística", logistic_model, X_train, X_test, y_train, y_test)

    # Entrena el modelo GaussianNB
    naive_bayes_model = train_naive_bayes(X_train, y_train)
    # Evalúa los resultados
    naive_bayes_results = evaluate_classifier("Gaussian Naive Bayes", naive_bayes_model, X_train, X_test, y_train, y_test)


    # Compara los resultados
    results = [logistic_results, naive_bayes_results]
    plot_roc_comparison(results)
    comparison = compare_models(results)

    return {
        "logistic_model": logistic_model,
        "naive_bayes_model": naive_bayes_model,
        "results": results,
        "comparison": comparison
    }


if __name__ == "__main__":

    # 1. Carga e inspección de datos
    print("\nCargando Breast Cancer Wisconsin Dataset...")
    df, cancer = load_data()

    # 2. Análisis exploratorio de datos
    explore_data(df, cancer)
    plot_class_distribution(df)
    plot_feature_histograms(df)
    target_correlation = plot_correlation_matrix(df)

    # 3. Entrenamiento, evaluación y comparación de clasificadores
    model_results = run_models(df)

    print("\nAnálisis completado.")
