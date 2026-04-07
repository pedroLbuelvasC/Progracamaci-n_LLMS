import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def generar_caso_de_uso_recomendar_reposicion():
    """
    Genera un caso de uso aleatorio (input, output) para la función
    recomendar_reposicion(df, k).

    Retorna:
        input_data  (dict): argumentos de la función solución.
        output_data (pd.DataFrame): DataFrame esperado con columnas
                    ['sku_id', 'cluster', 'z_seguridad', 'rop'].
    """
    rng = np.random.default_rng(seed=np.random.randint(0, 99999))

    # Parámetros aleatorios
    n_skus = rng.integers(40, 120)
    k = int(rng.integers(3, 6))  # entre 3 y 5 clústeres

    # Generar datos sintéticos
    sku_ids = [f"SKU-{i:04d}" for i in range(n_skus)]
    demanda_media = rng.uniform(10, 500, size=n_skus)
    demanda_std = demanda_media * rng.uniform(0.05, 0.40, size=n_skus)
    lead_time_dias = rng.uniform(3, 21, size=n_skus)
    costo_unidad = rng.uniform(1.5, 200.0, size=n_skus)
    stock_actual = rng.uniform(0, 1000, size=n_skus)

    df = pd.DataFrame({
        "sku_id": sku_ids,
        "demanda_media": demanda_media,
        "demanda_std": demanda_std,
        "lead_time_dias": lead_time_dias,
        "costo_unidad": costo_unidad,
        "stock_actual": stock_actual,
    })

    # ------------------------------------------------------------------ #
    # Calcular output esperado (ground truth)                              #
    # ------------------------------------------------------------------ #
    num_cols = ["demanda_media", "demanda_std", "lead_time_dias",
                "costo_unidad", "stock_actual"]
    X_num = df[num_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Promedio de demanda_media por clúster
    df_tmp = df.copy()
    df_tmp["cluster"] = clusters
    cluster_demand_mean = (
        df_tmp.groupby("cluster")["demanda_media"].mean()
    )

    max_cluster = int(cluster_demand_mean.idxmax())
    min_cluster = int(cluster_demand_mean.idxmin())

    def asignar_z(c):
        if c == max_cluster:
            return 2.05
        elif c == min_cluster:
            return 1.28
        else:
            return 1.65

    z_values = np.array([asignar_z(c) for c in clusters])

    # Cálculo vectorizado del ROP
    dm_diaria = df["demanda_media"].values / 7.0
    std_diaria = df["demanda_std"].values / np.sqrt(7.0)
    lt = df["lead_time_dias"].values

    rop = dm_diaria * lt + z_values * std_diaria * np.sqrt(lt)

    output_data = pd.DataFrame({
        "sku_id": df["sku_id"].values,
        "cluster": clusters.astype(int),
        "z_seguridad": z_values,
        "rop": rop,
    })
    output_data = output_data.sort_values("rop", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Ensamblar input                                                       #
    # ------------------------------------------------------------------ #
    input_data = {
        "df": df.copy(),
        "k": k,
    }

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_recomendar_reposicion()
    print("=== INPUT ===")
    print(f"DataFrame shape: {entrada['df'].shape}")
    print(f"k (clústeres)  : {entrada['k']}")
    print(entrada['df'].head(3))
    print("\n=== OUTPUT ESPERADO ===")
    print(f"Shape : {salida_esperada.shape}")
    print(f"Columnas: {list(salida_esperada.columns)}")
    print(salida_esperada.head(5))
