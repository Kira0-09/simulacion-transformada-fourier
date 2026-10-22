import numpy as np
import matplotlib.pyplot as plt

# 1. Configuracion de la simulacion
fs = 1000
T = 1.0
N = int(fs * T)
t = np.arange(N) / fs

# 2. Crear las señales
pulso = np.where((t >= 0.20) & (t < 0.30), 1.0, 0.0)
escalon = np.where(t >= 0.50, 1.0, 0.0)

f0 = 50
senoidal = np.sin(2 * np.pi * f0 * t)

# 3. Funcion para graficar cada señal y su FFT
def analizar_senal(x, nombre):
    X = np.fft.fft(x)
    frecuencias = np.fft.fftfreq(N, d=1/fs)

    Xc = np.fft.fftshift(X)
    fc = np.fft.fftshift(frecuencias)

    magnitud = np.abs(Xc) / N
    fase = np.angle(Xc)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    ax[0].plot(t, x)
    ax[0].set_title(nombre + " - Dominio del tiempo")
    ax[0].set_xlabel("Tiempo (s)")
    ax[0].set_ylabel("Amplitud")
    ax[0].grid(True)

    ax[1].plot(fc, magnitud)
    ax[1].set_title("Espectro de magnitud")
    ax[1].set_xlabel("Frecuencia (Hz)")
    ax[1].set_ylabel("Magnitud")
    ax[1].set_xlim(-200, 200)
    ax[1].grid(True)

    ax[2].plot(fc, fase)
    ax[2].set_title("Espectro de fase")
    ax[2].set_xlabel("Frecuencia (Hz)")
    ax[2].set_ylabel("Fase (rad)")
    ax[2].set_xlim(-200, 200)
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()

    return X

# 4. Analizar las tres señales
X_pulso = analizar_senal(pulso, "Pulso rectangular")
X_escalon = analizar_senal(escalon, "Funcion escalon")
X_senoidal = analizar_senal(senoidal, "Señal senoidal")

# 5. Comprobar la linealidad
x1 = pulso
x2 = 0.5 * senoidal

error_linealidad = np.max(np.abs(
    np.fft.fft(x1 + x2)
    - np.fft.fft(x1)
    - np.fft.fft(x2)
))

print("Error de linealidad:", error_linealidad)

# 6. Comprobar el desplazamiento temporal
desplazamiento = 100  # 100 muestras equivalen a 0.1 segundos

x_desplazada = np.roll(pulso, desplazamiento)

error_magnitud = np.max(np.abs(
    np.abs(np.fft.fft(pulso))
    - np.abs(np.fft.fft(x_desplazada))
))

print("Error de magnitud por desplazamiento:", error_magnitud)

# 7. Comprobar el escalamiento temporal
pulso_comprimido = np.where(
    (t >= 0.20) & (t < 0.25), 1.0, 0.0
)

plt.figure(figsize=(10, 4))
plt.plot(t, pulso, label="Pulso original")
plt.plot(t, pulso_comprimido, label="Pulso comprimido")
plt.xlim(0.15, 0.35)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Comparacion en el dominio del tiempo")
plt.legend()
plt.grid(True)
plt.show()

# Comparar los espectros
f = np.fft.fftfreq(N, d=1/fs)
f = np.fft.fftshift(f)

mag_original = np.abs(np.fft.fftshift(np.fft.fft(pulso))) / N
mag_comprimido = np.abs(
    np.fft.fftshift(np.fft.fft(pulso_comprimido))
) / N

plt.figure(figsize=(10, 4))
plt.plot(f, mag_original, label="Pulso original")
plt.plot(f, mag_comprimido, label="Pulso comprimido")
plt.xlim(-200, 200)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Comparacion de los espectros")
plt.legend()
plt.grid(True)
plt.show()
