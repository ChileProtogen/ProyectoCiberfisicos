"""
Sensor Virtual Físico-Informado (Estimador EKF para Coeficiente U(t)) - Proyecto S3
Universidad de Concepción - Análisis y Diseño de Sistemas Ciber-Físicos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def estimador_fisico_informado():
    # --- 1. CARGAR DATASET SINTÉTICO ---
    archivo_data = "data/dataset_ts_0.5_bits_10.csv"
    try:
        df = pd.read_csv(archivo_data)
    except FileNotFoundError:
        print(f"[Error] No se encontró {archivo_data}. Ejecuta data_generator.py primero.")
        return

    time = df['tiempo_s'].values
    T_measured = df['temperatura_medida_C'].values
    u_ctrl = df['potencia_u'].values
    T_amb = df['temperatura_ambiente_C'].values
    U_real = df['U_real_parametro'].values
    
    dt = time[1] - time[0]
    N = len(time)

    # --- 2. PARÁMETROS DEL MODELO FÍSICO ---
    C_th = 1200.0   # [J/K]
    A = 1.5         # [m^2]
    P_max = 500.0   # [W]

    # --- 3. INICIALIZACIÓN DEL FILTRO DE KALMAN EXTENDIDO (EKF) ---
    # Estado estimado x = [T_in, U]^T
    # Estimamos la temperatura y el parámetro U simultáneamente
    x_est = np.array([T_measured[0], 8.0]) 
    
    # Matriz de covarianza del error P
    P = np.diag([1.0, 1.0])
    
    # Ruido de proceso Q y de medición R
    Q = np.diag([1e-4, 1e-6]) # El parámetro U cambia muy lentamente (modelo de caminata aleatoria)
    R = 0.15                  # Varianza del ruido de medición (sensor + cuantización)

    U_est_history = np.zeros(N)
    T_est_history = np.zeros(N)

    # --- 4. BUCLE DE ESTIMACIÓN EN TIEMPO REAL ---
    for k in range(N):
        T_k, U_k = x_est[0], x_est[1]
        u_k = u_ctrl[k]
        T_a = T_amb[k]

        # A. PREDICCIÓN (Modelo Físico Continuo discretizado por Euler)
        # dT/dt = (u*P_max - U*A*(T - T_amb)) / C_th
        # dU/dt = 0
        dT_dt = (u_k * P_max - U_k * A * (T_k - T_a)) / C_th
        T_pred = T_k + dT_dt * dt
        U_pred = U_k # El parámetro es cuasi-estacionario

        x_pred = np.array([T_pred, U_pred])

        # Jacobiana del modelo de transición F = df/dx
        # d(dT/dt)/dT = - U*A / C_th
        # d(dT/dt)/dU = - A*(T - T_a) / C_th
        F = np.array([
            [1.0 + dt * (-U_k * A / C_th), dt * (-A * (T_k - T_a) / C_th)],
            [0.0,                         1.0                           ]
        ])

        P_pred = F @ P @ F.T + Q

        # B. ACTUALIZACIÓN (Corrección con la medición de temperatura T_measured)
        # Matriz de observación H (solo medimos T_in, no U directamente)
        H = np.array([[1.0, 0.0]])
        
        # Innovación (residuo)
        y_residual = T_measured[k] - H @ x_pred
        
        # Ganancia de Kalman
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T / S

        # Actualizar estado y covarianza
        x_est = x_pred + K.flatten() * y_residual
        P = (np.eye(2) - K @ H) @ P_pred

        U_est_history[k] = x_est[1]
        T_est_history[k] = x_est[0]

    # --- 5. CALCULO DE MÉTRICA (RMSE) ---
    from sklearn.metrics import mean_squared_error
    rmse_ekf = np.sqrt(mean_squared_error(U_real, U_est_history))
    print(f"[Resultado] RMSE del Modelo Físico-Informado (EKF): {rmse_ekf:.5f}")

    # --- 6. VISUALIZACIÓN COMPARATIVA ---
    plt.figure(figsize=(10, 5))
    plt.plot(time, U_real, label='Ground Truth $U(t)$ (Real)', color='black', linewidth=2, linestyle='--')
    plt.plot(time, U_est_history, label='Estimación Físico-Informada (EKF)', color='green', linewidth=2)
    
    plt.axvline(200.0, color='gray', linestyle=':', label='Inicio de Degradación ($t_0=200s$)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Coeficiente U [W/(m²·K)]')
    plt.title('Sensor Virtual Físico-Informado (EKF) vs. Parámetro Real de Degradación')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()

    import os
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/resultado_fisico_s3.png", dpi=300)
    print("[Éxito] Gráfico del modelo físico guardado en 'data/resultado_fisico_s3.png'.")
    plt.show()

if __name__ == '__main__':
    estimador_fisico_informado()