import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Path della serie temporale
path = "005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt"
# Rimuovi componenti anomale.
series = np.loadtxt(path)[0:4000]
# plotta la Serie temporale.
plt.plot(series)
plt.xlabel("Samples")
plt.ylabel("Values")
plt.show()
# Dividi in sottoseq

# Parametri per l'estrazione dei pattern
window_size = 100   
n_clusters = 10    
subsequences = []
for i in range(len(series) - window_size + 1):
    subsequences.append(series[i:i + window_size])

subsequences = np.array(subsequences)

kmeans = KMeans(n_clusters = n_clusters, random_state = 0)
labels = kmeans.fit_predict(subsequences)
for i in range(n_clusters):    
    cluster = subsequences[labels == i]
    plt.figure(figsize=(10, 6))
    for seq in cluster:
        plt.plot(seq, alpha=0.4)
    plt.title(f'Cluster {i+1}')
    plt.xlabel('Mese')
    plt.ylabel('Numero di passeggeri')
    plt.show()
