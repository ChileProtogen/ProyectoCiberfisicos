"""
Simulador de Planta Térmica con Degradación de Aislación (Hito S3)
Asignatura: Análisis y Diseño de Sistemas Ciber-Físicos
Universidad de Concepción
"""

import numpy as np
import matplotlib.pyplot as plt

def simular_planta_termica():
    # --- 1. PARÁMETROS DEL SISTEMA FÍSICO ---
    C_th = 1200.0   # Capacidad térmica [J/K]
    A = 1.5         # Área superficial efectiva [m^2]
    U_0 = 8.0       # Coeficiente de transferencia inicial [W/(m^2*K)]
    P_max = 500.0   # Potencia máxima del calefactor [W]
    T_amb = 20.0    # Temperatura ambiente [°C]

    # --- 2. PARÁMETROS DE DEGRADACIÓN ---
    t_0 = 200.0     # Instante en que inicia la degradación [s]
    alpha = 0.0003  # Tasa de degradación temporal [1/s]

    # --- 3. CONFIGURACIÓN DEL TIEMPO Y MUESTREO ---
    dt = 0.5        # Período de muestreo base / paso de integración [s]
    t_max = 600.0   # Tiempo total de simulación [s]
    time = np.arange(0, t_max + dt, dt)
    N = len(time)

    # Arreglos de almacenamiento
    T_in = np.zeros(N)
    U_coeff = np.zeros(N)
    u_ctrl = np.zeros(N)

    # Condición inicial
    T_in[0] = T_amb
    setpoint = 45.0  # Temperatura de consigna [°C]
    K_p = 8.0        # Ganancia del controlador proporcional

    # --- 4. BUCLE DE SIMULACIÓN (LAZO CERRADO) ---
    for k in range(1, N):
        t = time[k-1]
        
        # Modelo de degradación del parámetro no medido U(t)
        if t >= t_0:
            U_coeff[k-1] = U_0 * (1.0 + alpha * (t - t_0))
        else:
            U_coeff[k-1] = U_0
        
        # Ley de control (Acción proporcional acotada al rango [0, 1])
        error = setpoint - T_in[k-1]
        power_cmd = K_p * error
        u_ctrl[k-1] = np.clip(power_cmd / P_max, 0.0, 1.0)
        
        # Ecuación diferencial de la planta (Balance de energía térmico discretizado)
        dT_dt = (u_ctrl[k-1] * P_max - U_coeff[k-1] * A * (T_in[k-1] - T_amb)) / C_th
        T_in[k] = T_in[k-1] + dT_dt * dt

    # Asignar último valor para cierre de arreglos
    U_coeff[-1] = U_coeff[-2]
    u_ctrl[-1] = u_ctrl[-2]

    # --- 5. GENERACIÓN DE GRÁFICO DE EVIDENCIA (HITO S3) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Subplot 1: Temperatura y Control
    ax1.plot(time, T_in, label=r'Temperatura Interior $T_{in}(t)$', color='#d9534f', linewidth=2)
    ax1.axhline(setpoint, color='gray', linestyle='--', label='Setpoint (Consigna)')
    ax1.set_ylabel('Temperatura [°C]')
    ax1.set_title('Hito S3: Simulación de Planta Térmica y Degradación de Aislación')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='lower right')

    # Subplot 2: Degradación del Coeficiente U(t)
    ax2.plot(time, U_coeff, label=r'Coeficiente Real $U(t)$ (Degradado)', color='#0275d8', linewidth=2)
    ax2.axvline(t_0, color='#f0ad4e', linestyle=':', linewidth=2, label=f'Inicio de Degradación ($t_0={t_0}s$)')
    ax2.set_xlabel('Tiempo [s]')
    ax2.set_ylabel(r'U [W/(m²·K)]')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    
    # Guardar imagen para usar en el informe LaTeX
    output_filename = 'hito_s3_degradacion.png'
    plt.savefig(output_filename, dpi=300)
    print(f"[Éxito] Gráfico generado y guardado como '{output_filename}'.")
    
    plt.show()

if __name__ == '__main__':
    simular_planta_termica()