#! python

#
# Machine Learning I - Actividad grupal - Unidad 2
#
# Autor: Jaime Alberto Chica Betancourt
#
# Universidad de Investigación y Desarrollo – UDI
# Maestría en Ciencia de Datos e Inteligencia Artificial
# 7614 – Machine Learning I
# Docente: Jonathan Arley Monsalve Salazar
#
# Fecha de Entrega: 1 de Noviembre de 2026

"""
El script carga el dataset Breast Cancer Wisconsin y convierte el problema original en una clasificación de tres clases artificiales, según si mean radius es bajo, medio o alto. 


Después elimina mean radius de los predictores para no darle al modelo la respuesta directamente, divide los datos en entrenamiento y prueba y entrena cuatro clasificadores: 
    - Regresión Logística multinomial
    - Árbol de Decisión
    - Random Forest
    - Gradient Boosting.

Para los modelos basados en árboles busca además buenos hiperparámetros mediante GridSearchCV. 

Finalmente compara los cuatro modelos usando accuracy, F1 macro, F1 micro, matrices de confusión y AUC, analiza posibles diferencias entre entrenamiento y prueba y muestra qué variables resultan más importantes para los modelos basados en árboles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# Variable objetivo (la que queremos predecir)
TARGET_ORIGINAL = "target"

# Nombre de la nueva variable objetivo multiclase
TARGET_MULTICLASS = "radius_class"

# Variable utilizada para generar las clases artificiales
SOURCE_FEATURE = "mean radius"

# Proporción de observaciones utilizadas para prueba
TEST_SIZE = 0.20

# Aleatoriedad para obtener una partición reproducible, de acuerdo a la documentación de scikit-learn
RANDOM_STATE = 42

# Valores numéricos utilizados para representar las clases artificiales
CLASS_VALUES = [0, 1, 2]

# Nombres correspondientes a cada clase artificial
CLASS_NAMES = ["Radio bajo", "Radio medio", "Radio alto"]


def load_data():
    """
    Carga Breast Cancer Wisconsin como DataFrame.
    """
    cancer = load_breast_cancer(as_frame=True)
    return cancer.frame.copy(), cancer


def create_multiclass_target(df):

    """
    Crea tres clases aproximadamente equilibradas según mean radius, mostrando sus límites y distribución.
    """

    # Crea una copia del df
    result = df.copy()

    # Divide TARGET_MULTICLASS en 3 grupos (cuantiles) de tamaño aproximadamente similar
    result[TARGET_MULTICLASS] = pd.qcut(
        result[SOURCE_FEATURE],

        # Número de cuantiles
        q=3,

        # Asigna las clases 0 = Radio bajo, 1 = Radio medio y 2 = Radio alto (CLASS_VALUES)
        labels=CLASS_VALUES
    ).astype(int)

    # Agrupa las observaciones por clase y calcula cantidad, mínimo, máximo y promedio de mean radius para cada grupo
    summary = (
        result.groupby(TARGET_MULTICLASS, observed=True)[SOURCE_FEATURE]
        .agg(["count", "min", "max", "mean"])
        .rename(index=dict(zip(CLASS_VALUES, CLASS_NAMES)))
    )

    print("\nCLASES ARTIFICIALES SEGÚN MEAN RADIUS")
    print("=" * 75)
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))

    # Cuenta las observaciones de cada clase y sustituye los códigos numéricos por nombres descriptivos
    distribution = (
        result[TARGET_MULTICLASS]
        .value_counts()
        .sort_index()
        .rename(index=dict(zip(CLASS_VALUES, CLASS_NAMES)))
    )

    print("\nDistribución de clases:")
    print(distribution)

    # Grafica la distribución para comprobar visualmente el balance entre las tres clases artificiales
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=CLASS_NAMES, y=distribution.values, hue=CLASS_NAMES, palette="Set2", legend=False)

    for index, value in enumerate(distribution.values):
        ax.text(index, value + 3, str(value), ha="center")

    plt.title("Distribución de las clases artificiales")
    plt.xlabel("Clase")
    plt.ylabel("Número de observaciones")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    return result


def train_logistic_regression(X_train, y_train):
    
    """
    Entrena la Regresión Logística multinomial con datos estandarizados.
    """

    # Estandariza los predictores y  entrena el modelo de Regresión Logística
    model = Pipeline([
        ("scaler", StandardScaler()),

        # Con tres clases, el solver lbfgs aplica automáticamente la formulación multinomial en las versiones actuales de scikit-learn
        ("model", LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_STATE
        ))
    ])

    # Ajusta el escalador y entrena el modelo utilizando únicamente los datos de entrenamiento.
    model.fit(X_train, y_train)

    return model, None


def tune_decision_tree(X_train, y_train):

    """
    Optimiza un Árbol de Decisión mediante GridSearchCV.
    """

    # Crea el Árbol de Decisión que será optimizado
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)

    # Define los valores que se probarán para controlar la profundidad y complejidad del árbol
    hyperparameters = {
        "max_depth": [3, 5, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    # Prueba las combinaciones mediante validación cruzada y selecciona la que obtiene el mayor F1 macro
    search = GridSearchCV(
        estimator=model,
        param_grid=hyperparameters,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1
    )
    
    # Ejecuta la búsqueda utilizando únicamente los datos de entrenamiento
    search.fit(X_train, y_train)

    # Devuelve el mejor modelo y los hiperparámetros seleccionados
    return search.best_estimator_, search.best_params_


def tune_random_forest(X_train, y_train):

    """
    Busca la mejor configuración de Random Forest mediante GridSearchCV.
    """

    # Crea el clasificador Random Forest base
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    # Define los valores de los hiperparámetros que serán evaluados
    hyperparameters = {
        "n_estimators": [100, 200],
        "max_depth": [5, None],
        "min_samples_leaf": [1, 2]
    }

    # Prueba todas las combinaciones mediante validación cruzada y selecciona la que obtiene el mejor F1 macro
    search = GridSearchCV(
        estimator=model,
        param_grid=hyperparameters,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1
    )
    
    # Ejecuta la búsqueda utilizando únicamente los datos de entrenamiento
    search.fit(X_train, y_train)
    
    # Devuelve el mejor modelo y los hiperparámetros seleccionados
    return search.best_estimator_, search.best_params_


def tune_gradient_boosting(X_train, y_train):

    """
    Busca la mejor configuración de Gradient Boosting mediante GridSearchCV
    """

    # Crea el clasificador Gradient Boosting base
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)

    # Define los valores de los hiperparámetros que serán evaluados
    hyperparameters = {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.10],
        "max_depth": [1, 2]
    }

    # Prueba todas las combinaciones mediante validación cruzada y selecciona la que obtiene el mejor F1 macro
    search = GridSearchCV(
        estimator=model,
        param_grid=hyperparameters,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1
    )

    # Ejecuta la búsqueda utilizando únicamente los datos de entrenamiento
    search.fit(X_train, y_train)

    # Devuelve el mejor modelo y los hiperparámetros seleccionados
    return search.best_estimator_, search.best_params_


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):

    """
    Calcula métricas, matriz de confusión y probabilidades multiclase.
    """

    # Genera las predicciones y probabilidades para evaluar el modelo
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_probability = model.predict_proba(X_test)

    # Calcula accuracy y F1 macro/micro
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    f1_macro = f1_score(y_test, y_test_pred, average="macro")
    f1_micro = f1_score(y_test, y_test_pred, average="micro")

    # Construye la matriz de confusión en el orden bajo, medio y alto
    matrix = confusion_matrix(y_test, y_test_pred, labels=CLASS_VALUES)

    # Convierte las clases al formato One-vs-Rest y calcula el AUC macro dando la misma importancia a cada clase
    y_test_binary = label_binarize(y_test, classes=CLASS_VALUES)
    auc_macro = roc_auc_score(y_test_binary, y_test_probability, average="macro", multi_class="ovr")

    # Calcula una curva ROC para cada clase y las interpola sobre una rejilla común para obtener una curva macro promedio
    mean_fpr = np.linspace(0, 1, 200)
    interpolated_tpr = []
    roc_by_class = {}

    for class_index, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(
            y_test_binary[:, class_index],
            y_test_probability[:, class_index]
        )
        interpolated_tpr.append(np.interp(mean_fpr, fpr, tpr))
        roc_by_class[class_name] = {"FPR": fpr, "TPR": tpr}

    mean_tpr = np.mean(interpolated_tpr, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    print(f"\nResultados - {model_name}")
    print("-" * 60)
    print(f"Accuracy Train : {accuracy_train:.4f}")
    print(f"Accuracy Test  : {accuracy_test:.4f}")
    print(f"F1 Macro       : {f1_macro:.4f}")
    print(f"F1 Micro       : {f1_micro:.4f}")
    print(f"AUC Macro OVR  : {auc_macro:.4f}")
    print("Matriz de confusión [filas=reales, columnas=predichas]:")
    print(matrix)

    return {
        "Model": model_name,
        "Accuracy Train": accuracy_train,
        "Accuracy Test": accuracy_test,
        "F1 Macro": f1_macro,
        "F1 Micro": f1_micro,
        "AUC Macro OVR": auc_macro,
        "Confusion Matrix": matrix,
        "Mean FPR": mean_fpr,
        "Mean TPR": mean_tpr,
        "ROC by Class": roc_by_class
    }


def plot_confusion_matrices(results):
    
    """
    Grafica las matrices de confusión de los cuatro modelos.
    """

    figure, axes = plt.subplots(2, 2, figsize=(13, 11))
    for axis, result in zip(axes.ravel(), results):
        sns.heatmap(
            result["Confusion Matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=axis
        )
        axis.set_title(result["Model"])
        axis.set_xlabel("Clase predicha")
        axis.set_ylabel("Clase real")

    plt.suptitle("Matrices de confusión multiclase", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_comparison(results):

    """
    Compara las curvas ROC macro One-vs-Rest de todos los modelos.
    """

    plt.figure(figsize=(9, 7))
    for result in results:
        plt.plot(
            result["Mean FPR"],
            result["Mean TPR"],
            linewidth=2,
            label=(
                f'{result["Model"]} '
                f'(AUC macro = {result["AUC Macro OVR"]:.4f})'
            )
        )

    plt.plot([0, 1], [0, 1], "k--", label="Clasificador aleatorio")
    plt.title("Comparación ROC multiclase - promedio macro One-vs-Rest")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(models, feature_names):

    """
    Grafica las diez variables más importantes de los modelos de árboles.
    """
        
    tree_models = {
        name: model
        for name, model in models.items()
        if hasattr(model, "feature_importances_")
    }

    figure, axes = plt.subplots(1, len(tree_models), figsize=(18, 6))
    if len(tree_models) == 1:
        axes = [axes]

    for axis, (model_name, model) in zip(axes, tree_models.items()):
        importance = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False).head(10).sort_values()

        importance.plot(kind="barh", ax=axis, color="#4C72B0")
        axis.set_title(model_name)
        axis.set_xlabel("Importancia")
        axis.set_ylabel("Variable")

    plt.suptitle("Diez variables más importantes por modelo", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def compare_models(results):

    """
    Construye la tabla resumen e identifica los mejores resultados.
    """

    # Construye una tabla con las principales métricas obtenidas por cada modelo evaluado
    comparison = pd.DataFrame([
        {
            "Model": result["Model"],
            "Accuracy Train": result["Accuracy Train"],
            "Accuracy Test": result["Accuracy Test"],
            "F1 Macro": result["F1 Macro"],
            "F1 Micro": result["F1 Micro"],
            "AUC Macro OVR": result["AUC Macro OVR"]
        }
        for result in results
    ])

    # Calcula la diferencia entre el accuracy de entrenamiento y de prueba como apoyo para analizar la generalización
    comparison["Accuracy Gap"] = (comparison["Accuracy Train"] - comparison["Accuracy Test"]).abs()

    print("\nCOMPARACIÓN DE MODELOS")
    print("=" * 100)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


    # Modelo con mayor accuracy
    max_accuracy = comparison["Accuracy Test"].max()
    best_accuracy_models = comparison.loc[np.isclose(comparison["Accuracy Test"], max_accuracy), "Model"].tolist()
    print("\nMayor accuracy de prueba:\n" f"{', '.join(best_accuracy_models)} ({max_accuracy:.4f})")

    # Modelo con mayor F1 macro
    max_f1 = comparison["F1 Macro"].max()
    best_f1_models = comparison.loc[np.isclose(comparison["F1 Macro"], max_f1), "Model"].tolist()
    print("Mayor F1 macro:\n" f"{', '.join(best_f1_models)} ({max_f1:.4f})")

    # Modelo con mayor AUC macro
    best_auc = comparison.loc[comparison["AUC Macro OVR"].idxmax(), "Model"]
    print(f"Mayor AUC macro OVR: {best_auc}")

    return comparison


def run_models(df):

    """
    Prepara los datos, entrena y optimiza los cuatro clasificadores, genera las visualizaciones y compara sus resultados
    """

    # Separa los predictores de la variable objetivo y elimina las variables que producirían fuga de información.  Se elimina TARGET_ORIGINAL porque contiene el diagnóstico original, así como TARGET_MULTICLASS.  También se elimina SOURCE_FEATURE porque esta variable creó las clases y dejarla en X produciría fuga directa de información.
    X = df.drop(columns=[TARGET_ORIGINAL, TARGET_MULTICLASS, SOURCE_FEATURE])
    y = df[TARGET_MULTICLASS]

    # Divide los datos manteniendo la proporción de las tres clases
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    print("\nDISTRIBUCIÓN DESPUÉS DE LA PARTICIÓN")
    print("=" * 75)
    print("Entrenamiento:")
    print(y_train.value_counts().sort_index())
    print("Prueba:")
    print(y_test.value_counts().sort_index())

    models = {}
    hyperparameters = {}

    # Entrena la Regresión Logística y optimiza los modelos de árboles
    print("\nEntrenando Regresión Logística multinomial...")
    models["Regresión logística"] , hyperparameters["Regresión logística"] = (train_logistic_regression(X_train, y_train))

    print("Optimizando Árbol de decisión...")
    models["Árbol de decisión"], hyperparameters["Árbol de decisión"] = (tune_decision_tree(X_train, y_train))

    print("Optimizando Random Forest...")
    models["Random Forest"], hyperparameters["Random Forest"] = (tune_random_forest(X_train, y_train))

    print("Optimizando Gradient Boosting...")
    models["Gradient Boosting"], hyperparameters["Gradient Boosting"] = (tune_gradient_boosting(X_train, y_train))

    print("\nMEJORES HIPERPARÁMETROS")
    print("=" * 75)
    for model_name, best_hyperparameters in hyperparameters.items():
        print(f"{model_name}: {best_hyperparameters}")

    results = []
    
    # Evalúa todos los modelos utilizando los mismos datos
    for model_name, model in models.items():
        results.append(evaluate_model(model_name, model, X_train, X_test, y_train, y_test))

    # Genera las visualizaciones y la tabla comparativa
    plot_confusion_matrices(results)
    plot_roc_comparison(results)
    plot_feature_importance(models, X_train.columns)

    # Compara los modelos
    comparison = compare_models(results)

    return {
        "models": models,
        "best_hyperparameters": hyperparameters,
        "results": results,
        "comparison": comparison
    }


if __name__ == "__main__":

    # 1. Carga e inspección de datos
    print("\nCargando Breast Cancer Wisconsin Dataset...")
    data, cancer = load_data()

    # 2. Crea la clasificación de clases artificiales
    data = create_multiclass_target(data)

    # 3. Entrenamiento, evaluación y comparación de clasificadores
    model_results = run_models(data)
    print("\nAnálisis completado.")
