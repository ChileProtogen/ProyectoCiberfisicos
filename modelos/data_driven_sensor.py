"""
Sensor Virtual Basado en Datos (MLP Regressor) - Proyecto S3
Universidad de Concepción - Análisis y Diseño de Sistemas Ciber-Físicos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def entrenar_y_evaluar_mlp():
    # --- 1. CARGAR DATASET SINTÉTICO ---
    # Asegúrate de haber corrido data_generator.py previamente
    archivo_data = "data/dataset_ts_0.5_bits_10.csv"
    try:
        df = pd.read_csv(archivo_data)
    except FileNotFoundError:
        print(f"[Error] No se encontró el archivo {archivo_data}. Ejecuta primero data_generator.py")
        return

    # Características de entrada (Variables observables en el CPS)
    # X: Temperatura medida, Potencia de control, Temperatura ambiente
    X = df[['temperatura_medida_C', 'potencia_u', 'temperatura_ambiente_C']].values
    
    # Variable objetivo (Parámetro no medido que queremos estimar con el sensor virtual)
    y = df['U_real_parametro'].values

    # --- 2. DIVISIÓN ENTRENAMIENTO / PRUEBA (Split temporal o aleatorio) ---
    # Usamos los primeros 400s para entrenar y los últimos 200s (zona de degradación avanzada) para probar
    split_idx = int(len(df) * 0.65)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    t_train, t_test = df['tiempo_s'].values[:split_idx], df['tiempo_s'].values[split_idx:]

    # --- 3. NORMALIZACIÓN DE DATOS (Crucial para redes neuronales) ---
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # --- 4. ENTRENAMIENTO DEL MODELO BASADO EN DATOS (MLP) ---
    print("[Modelo] Entrenando Red Neuronal Multicapa (MLP)...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)

    # --- 5. INFERENCIA Y EVALUACIÓN ---
    y_pred_train = mlp.predict(X_train_scaled)
    y_pred_test = mlp.predict(X_test_scaled)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"[Resultado] RMSE en Entrenamiento: {rmse_train:.5f}")
    print(f"[Resultado] RMSE en Prueba (Test): {rmse_test:.5f}")

    # --- 6. VISUALIZACIÓN COMPARATIVA ---
    plt.figure(figsize=(10, 5))
    plt.plot(df['tiempo_s'], y, label='Ground Truth $U(t)$ (Real)', color='black', linewidth=2, linestyle='--')
    plt.plot(t_train, y_pred_train, label='Estimación MLP (Train)', color='blue', alpha=0.7)
    plt.plot(t_test, y_pred_test, label='Estimación MLP (Test)', color='red', linewidth=2)
    
    plt.axvline(df['tiempo_s'].values[split_idx], color='gray', linestyle=':', label='Corte Train/Test')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Coeficiente U [W/(m²·K)]')
    plt.title('Sensor Virtual Basado en Datos (MLP) vs. Parámetro Real de Degradación')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Guardar gráfico de resultados del modelo
    import os
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/resultado_mlp_s3.png", dpi=300)
    print("[Éxito] Gráfico de desempeño MLP guardado en 'data/resultado_mlp_s3.png'.")
    plt.show()

if __name__ == '__main__':
    entrenar_y_evaluar_mlp()