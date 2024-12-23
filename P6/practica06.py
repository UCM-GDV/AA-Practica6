import numpy as np
import matplotlib.pyplot as plt
from Utils import load_data_csv
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from kneed import KneeLocator

# Ejercicio 2
np.random.seed(0)

data, X, y = load_data_csv("KartData.csv", ["ray1", "ray2", "ray3", "ray4", "ray5", "kartx", "karty", "kartz", "time"], "action")

# silhouette_coefficients = []
inertia_values = []
for k in range(2, 9):
    kkkmeans = KMeans(n_clusters=k)
    kkkmeans.fit(X)
    # score = silhouette_score(X, kkkmeans.labels_)
    # silhouette_coefficients.append(score)
    inertia_values.append(kkkmeans.inertia_)

fig, ax = plt.subplots(figsize = (24, 7))
# ax.scatter(range(2, 9), silhouette_coefficients)
ax.scatter(range(2, 9), inertia_values)
ax.set_xticks(range(2, 9))
ax.set_xlabel("Número de clusters")
# ax.set_ylabel("Promedio coeficientes de Silhouette")
ax.set_ylabel("SSE / Valores de inercia")

kl = KneeLocator(range(2, 9), inertia_values, curve="convex", direction="decreasing")
print(kl.knee)

plt.grid()
plt.savefig("ejercicio2.png")
plt.show()