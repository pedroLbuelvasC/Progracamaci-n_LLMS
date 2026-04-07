import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def generar_caso_de_uso_clasificar_falla_electrica():
    """
    Genera un caso de uso aleatorio (input, output) para la función
    clasificar_falla_electrica(X_train, y_train, X_test, n_components).

    Retorna:
        input_data  (dict): argumentos de la función solución.
        output_data (np.ndarray): predicciones esperadas sobre X_test.
    """
    rng = np.random.default_rng(seed=np.random.randint(0, 99999))

    # Parámetros aleatorios
    n_total = rng.integers(200, 400)
    n_features = rng.integers(10, 20)
    n_components = int(rng.integers(4, min(8, n_features - 1)))
    test_fraction = float(rng.choice([0.2, 0.25, 0.3]))

    # Número de clases: 4 tipos de falla
    n_classes = 4
    # Distribución desbalanceada (la falla 0 es más común)
    class_weights = np.array([0.5, 0.2, 0.2, 0.1])

    # Generar datos sintéticos con separabilidad moderada
    y_all = rng.choice(n_classes, size=n_total, p=class_weights)

    X_all = np.zeros((n_total, n_features))
    for cls in range(n_classes):
        mask = y_all == cls
        n_cls = mask.sum()
        # Cada clase tiene una media diferente en el espacio de características
        center = rng.uniform(-3, 3, size=n_features) * cls
        X_all[mask] = rng.normal(loc=center, scale=1.5, size=(n_cls, n_features))

    # Inyectar outliers (valores de impulso eléctrico)
    n_outliers = max(5, int(n_total * 0.04))
    outlier_idx = rng.choice(n_total, size=n_outliers, replace=False)
    X_all[outlier_idx] += rng.uniform(10, 30, size=(n_outliers, n_features))

    # Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=test_fraction,
        random_state=42,
        stratify=y_all,
    )

    # ------------------------------------------------------------------ #
    # Calcular output esperado (ground truth)                              #
    # ------------------------------------------------------------------ #
    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    output_data = pipe.predict(X_test)

    # ------------------------------------------------------------------ #
    # Ensamblar input                                                       #
    # ------------------------------------------------------------------ #
    input_data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "n_components": n_components,
    }

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_clasificar_falla_electrica()
    print("=== INPUT ===")
    print(f"X_train shape  : {entrada['X_train'].shape}")
    print(f"y_train shape  : {entrada['y_train'].shape}")
    print(f"X_test shape   : {entrada['X_test'].shape}")
    print(f"n_components   : {entrada['n_components']}")
    print(f"Clases en train: {sorted(set(entrada['y_train'].tolist()))}")
    print("\n=== OUTPUT ESPERADO ===")
    print(f"Shape      : {salida_esperada.shape}")
    print(f"Dtype      : {salida_esperada.dtype}")
    print(f"Predicciones (primeras 10): {salida_esperada[:10]}")
    unique, counts = np.unique(salida_esperada, return_counts=True)
    print(f"Distribución: {dict(zip(unique.tolist(), counts.tolist()))}")
