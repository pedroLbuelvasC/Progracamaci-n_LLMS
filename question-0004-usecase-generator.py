import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def generar_caso_de_uso_predecir_degradacion_bateria():
    """
    Genera un caso de uso aleatorio (input, output) para la función
    predecir_degradacion_bateria(df_train, df_test).

    Retorna:
        input_data  (dict): argumentos de la función solución.
        output_data (pd.DataFrame): DataFrame con columnas ['ciclo', 'soh_pred'].
    """
    rng = np.random.default_rng(seed=np.random.randint(0, 99999))

    # Parámetros aleatorios
    n_train = rng.integers(120, 300)
    n_test = rng.integers(20, 60)

    def _generar_ciclos(n, start_ciclo=1, incluir_soh=True, rng=rng):
        ciclos = np.arange(start_ciclo, start_ciclo + n)

        # SoH decrece con el ciclo + ruido gaussiano
        degradacion = 100 - (ciclos / (ciclos.max() + 50)) * 25
        soh = degradacion + rng.normal(0, 0.5, size=n)
        soh = np.clip(soh, 70, 100)

        temp = rng.uniform(20, 60, size=n)
        voltaje = rng.uniform(2.5, 4.2, size=n)
        corriente = rng.uniform(0.5, 5.0, size=n)
        tiempo_carga = rng.uniform(30, 120, size=n)
        resistencia = rng.uniform(50, 200, size=n) + (ciclos * 0.05)

        data = {
            "ciclo": ciclos.astype(int),
            "temp_maxima_C": temp,
            "voltaje_final_V": voltaje,
            "corriente_media_A": corriente,
            "tiempo_carga_min": tiempo_carga,
            "resistencia_interna": resistencia,
        }
        if incluir_soh:
            data["soh_pct"] = soh

        return pd.DataFrame(data)

    df_train_raw = _generar_ciclos(n_train, start_ciclo=1, incluir_soh=True)
    df_test_raw = _generar_ciclos(n_test, start_ciclo=n_train + 1, incluir_soh=False)

    # Inyectar anomalías en train para que la limpieza tenga efecto
    n_nulos = rng.integers(3, 8)
    idx_nulos = rng.choice(n_train, size=n_nulos, replace=False)
    col_nula = rng.choice(["temp_maxima_C", "corriente_media_A", "tiempo_carga_min"])
    df_train_raw.loc[idx_nulos, col_nula] = np.nan

    n_volt_neg = rng.integers(2, 5)
    idx_volt_neg = rng.choice(n_train, size=n_volt_neg, replace=False)
    df_train_raw.loc[idx_volt_neg, "voltaje_final_V"] = rng.uniform(-1, 0, size=n_volt_neg)

    n_temp_alta = rng.integers(2, 5)
    idx_temp_alta = rng.choice(n_train, size=n_temp_alta, replace=False)
    df_train_raw.loc[idx_temp_alta, "temp_maxima_C"] = rng.uniform(81, 120, size=n_temp_alta)

    # ------------------------------------------------------------------ #
    # Calcular output esperado (ground truth)                              #
    # ------------------------------------------------------------------ #
    # Limpieza de train
    df_clean = df_train_raw.dropna()
    df_clean = df_clean[df_clean["voltaje_final_V"] > 0]
    df_clean = df_clean[df_clean["temp_maxima_C"] <= 80]

    feature_cols = ["temp_maxima_C", "voltaje_final_V", "corriente_media_A",
                    "tiempo_carga_min", "resistencia_interna"]

    X_train = df_clean[feature_cols].values
    y_train = df_clean["soh_pct"].values
    X_test = df_test_raw[feature_cols].values

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    output_data = pd.DataFrame({
        "ciclo": df_test_raw["ciclo"].values,
        "soh_pred": np.round(y_pred, 4),
    }).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Ensamblar input                                                       #
    # ------------------------------------------------------------------ #
    input_data = {
        "df_train": df_train_raw.copy(),
        "df_test": df_test_raw.copy(),
    }

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_predecir_degradacion_bateria()
    print("=== INPUT ===")
    print(f"df_train shape: {entrada['df_train'].shape}")
    print(f"df_test shape : {entrada['df_test'].shape}")
    print(f"Columnas train: {list(entrada['df_train'].columns)}")
    print(f"Columnas test : {list(entrada['df_test'].columns)}")
    print(f"Nulos en train por columna:\n{entrada['df_train'].isnull().sum()}")
    print("\n=== OUTPUT ESPERADO ===")
    print(f"Shape   : {salida_esperada.shape}")
    print(f"Columnas: {list(salida_esperada.columns)}")
    print(salida_esperada.head(8))
