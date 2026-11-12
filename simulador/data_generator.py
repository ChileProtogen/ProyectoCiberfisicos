"""
Generador de Dataset Sintético con Perturbaciones Ciber-Físicas (Proyecto S3)
Universidad de Concepción - Análisis y Diseño de Sistemas Ciber-Físicos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cuantizar_senal(signal, bits):
    """Simula la cuantización de un conversor Analógico-Digital (ADC) de b bits."""
    n_niveles = 2 ** bits
    s_min, s_max = np.min(signal), np.max(signal)
    if s_max == s_min:
        return signal
    # Normalizar a [0, 1], cuantizar y desnormalizar
    normalizada = (signal - s_min) / (s_max - s_min)
    cuantizada = np.round(normalizada * (n_niveles - 1)) / (n_niveles - 1)
    return cuantizada * (s_max - s_min) + s_min

def generar_dataset_s3(ts=0.5, bits_adc=10, ruido_std=0.2, guardar_csv=True):
    # --- Parámetros físicos ---
    C_th = 1200.0   # Capacidad térmica [J/K]
    A = 1.5         # Área superficial [m^2]
    U_0 = 8.0       # Coeficiente inicial [W/(m^2*K)]
    P_max = 500.0   # Potencia máxima [W]
    T_amb = 20.0    # Temperatura ambiente [°C]

    # --- Degradación ---
    t_0 = 200.0     
    alpha = 0.0003  

    # --- Tiempo ---
    t_max = 600.0
    time = np.arange(0, t_max + ts, ts)
    N = len(time)

    T_in = np.zeros(N)
    U_real = np.zeros(N)
    u_ctrl = np.zeros(N)

    T_in[0] = T_amb
    setpoint = 45.0
    K_p = 8.0

    # Simulación en lazo cerrado
    for k in range(1, N):
        t = time[k-1]
        U_real[k-1] = U_0 * (1.0 + alpha * (t - t_0)) if t >= t_0 else U_0
        
        error = setpoint - T_in[k-1]
        u_ctrl[k-1] = np.clip((K_p * error) / P_max, 0.0, 1.0)
        
        dT_dt = (u_ctrl[k-1] * P_max - U_real[k-1] * A * (T_in[k-1] - T_amb)) / C_th
        T_in[k] = T_in[k-1] + dT_dt * ts

    U_real[-1] = U_real[-2]
    u_ctrl[-1] = u_ctrl[-2]

    # Aplicar perturbaciones ciber-físicas (Ruido y Cuantización ADC)
    np.random.seed(42) # Reproducibilidad
    ruido = np.random.normal(0, ruido_std, size=N)
    T_in_ruido = T_in + ruido
    T_in_cuantizada = cuantizar_senal(T_in_ruido, bits=bits_adc)

    # Consolidar en un DataFrame
    df = pd.DataFrame({
        'tiempo_s': time,
        'temperatura_ideal_C': T_in,
        'temperatura_medida_C': T_in_cuantizada,
        'potencia_u': u_ctrl,
        'temperatura_ambiente_C': T_amb,
        'U_real_parametro': U_real # Ground truth para entrenar/evaluar el sensor virtual
    })

    if guardar_csv:
        nombre_archivo = f"data/dataset_ts_{ts}_bits_{bits_adc}.csv"
        df.to_csv(nombre_archivo, index=False)
        print(f"[Dataset] Generado y guardado en '{nombre_archivo}'")

    return df

if __name__ == '__main__':
    # Generar dataset base con Ts=0.5s, ADC de 10 bits y ruido moderado
    df_ejemplo = generar_dataset_s3(ts=0.5, bits_adc=10, ruido_std=0.15)