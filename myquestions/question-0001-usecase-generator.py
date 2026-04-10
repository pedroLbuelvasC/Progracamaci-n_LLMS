import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def generar_caso_de_uso_detectar_vibracion_anomala():
    """
    Genera un caso de uso aleatorio (input, output) para la función
    detectar_vibracion_anomala(df, n_components, umbral_percentil).

    Retorna:
        input_data  (dict): argumentos de la función solución.
        output_data (np.ndarray): array booleano esperado.
    """
    rng = np.random.default_rng(seed=np.random.randint(0, 99999))

    # Dimensiones aleatorias
    n_ventanas = rng.integers(60, 150)        # número de filas
    n_muestras = rng.integers(20, 50)         # columnas de señal
    n_components = int(rng.integers(2, min(8, n_muestras // 2)))
    umbral_percentil = float(rng.choice([90, 92, 95, 97]))
    n_anomalas = rng.integers(5, 15)

    # ------------------------------------------------------------------ #
    # Generar señal "normal": variaciones suaves de baja amplitud         #
    # ------------------------------------------------------------------ #
    X_normal = rng.normal(loc=0.0, scale=0.5, size=(n_ventanas, n_muestras))

    # Inyectar ventanas anómalas con alta amplitud en posiciones aleatorias
    idx_anomalos = rng.choice(n_ventanas, size=n_anomalas, replace=False)
    X_normal[idx_anomalos] += rng.normal(
        loc=0.0, scale=5.0, size=(n_anomalas, n_muestras)
    )

    # Construir DataFrame con nombres de columnas y columna categórica
    col_names = [f"v{i}" for i in range(n_muestras)]
    df = pd.DataFrame(X_normal, columns=col_names)

    # Añadir columna categórica que debe ser ignorada
    n_motores = rng.integers(2, 5)
    df["motor_id"] = rng.choice(
        [f"MOT-{k:02d}" for k in range(n_motores)], size=n_ventanas
    )

    # ------------------------------------------------------------------ #
    # Calcular output esperado (ground truth)                              #
    # ------------------------------------------------------------------ #
    X_num = df.select_dtypes(include=[np.number]).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_reduced)

    # Error cuadrático medio por fila (sin loop)
    errores = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

    umbral = np.percentile(errores, umbral_percentil)
    output_data = errores > umbral

    # ------------------------------------------------------------------ #
    # Ensamblar input                                                       #
    # ------------------------------------------------------------------ #
    input_data = {
        "df": df.copy(),
        "n_components": n_components,
        "umbral_percentil": umbral_percentil,
    }

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_detectar_vibracion_anomala()
    print("=== INPUT ===")
    print(f"DataFrame shape : {entrada['df'].shape}")
    print(f"Columnas        : {list(entrada['df'].columns)}")
    print(f"n_components    : {entrada['n_components']}")
    print(f"umbral_percentil: {entrada['umbral_percentil']}")
    print("\n=== OUTPUT ESPERADO ===")
    print(f"Tipo  : {type(salida_esperada)}")
    print(f"Shape : {salida_esperada.shape}")
    print(f"Dtype : {salida_esperada.dtype}")
    print(f"Anomalías detectadas: {salida_esperada.sum()} / {len(salida_esperada)}")
