import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

# Exemplo de cálculo da vida útil de fadiga usando a Equação de Miner
S = np.array([100, 150, 200])  # Tensão aplicada em cada ciclo (MPa)
N = np.array([1e5, 1e6, 1e7])  # Número de ciclos em cada nível de tensão
S_e = 200  # Tensão de resistência à fadiga (MPa)
k = 0.5  # Fator de Miner
n = len(S)  # Número de níveis de tensão

# Calcular o dano acumulado em cada nível de tensão
D = np.zeros(n)
for i in range(n):
    D[i] = N[i] * (S[i] / S_e)**k

# Calcular o dano total acumulado
D_total = np.sum(D)

# Calcular a vida útil de fadiga
N_f = (1 / D_total)**(1 / k)

print(f"Vida útil de fadiga: {N_f:.2e} ciclos")

# Exemplo de ajuste de distribuição de probabilidade aos dados de fadiga
data = pd.read_csv('dados_fadiga.csv', header=None)
data.columns = ['Amostra', 'Falha']
falhas = data['Falha'].values

# Ajustar distribuição Weibull aos dados
shape, loc, scale = stats.weibull_min.fit(falhas, floc=0)
print(f"Parâmetros da distribuição Weibull: shape={shape:.2f}, loc={loc:.2f}, scale={scale:.2f}")

# Calcular a probabilidade de falha em 1e6 ciclos
prob_falha = 1 - stats.weibull_min.cdf(1e6, shape, loc=loc, scale=scale)
print(f"Probabilidade de falha em 1e6 ciclos: {prob_falha:.2%}")

# Plotar gráfico de S-N
S_n = np.array([50, 100, 150, 200, 250, 300])  # Tensão aplicada (MPa)
N_f = np.array([1e6, 5e6, 1e7, 2e7, 3e7, 5e7])  # Vida útil de fadiga correspondente
plt.plot(np.log10(N_f), np.log10(S_n))
plt.xlabel('Log10(Número de ciclos)')
plt.ylabel('Log10(Tensão aplicada)')
plt.title('Gráfico de S-N')
plt.show()
