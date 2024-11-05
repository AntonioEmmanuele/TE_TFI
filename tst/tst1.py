import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset "flights" di seaborn
data = sns.load_dataset('flights')
data['month'] = pd.to_datetime(data['month'], format='%b').dt.month  # Converte il mese in numerico
data = data.sort_values(['year', 'month'])
data.set_index(pd.to_datetime(data[['year', 'month']].assign(day=1)), inplace=True)

# Estrai la serie temporale dei passeggeri
series = data['passengers'].values
plt.plot(series)
plt.xlabel('Mese')
plt.ylabel('Numero di passeggeri')
plt.title("Serie originale")
plt.show()

# Parametri per l'estrazione dei pattern
window_size = 12  # Lunghezza delle sotto-sequenze (qui per vedere pattern annuali)
n_clusters = 3    # Numero di cluster

# Creazione delle sotto-sequenze, le sottosequenze vengono generate tramite una semplice sliding window.
subsequences = []
for i in range(len(series) - window_size + 1):
    subsequences.append(series[i:i + window_size])

# Converti le sotto-sequenze in un array numpy
subsequences = np.array(subsequences)

# Applica il clustering sulle sotto-sequenze
kmeans = KMeans(n_clusters = n_clusters, random_state = 0)
# Per ogni possibile sottosequenza ritorna l'indice del cluster in cui
# viene inserita. La cosa che puoi facilmente notare è che per come è semplice il dataset
# Allora il cluster mette le prime sottosequenze in un unico cluster e poi le altre in un secondo
labels = kmeans.fit_predict(subsequences)
# Visualizza i cluster
for i in range(n_clusters):
    # print("Subsequences are ")
    # print(subsequences)
    # print("Cluster")
    cluster = subsequences[labels == i]
    # print(cluster)
    # exit(1)
    plt.figure(figsize=(10, 6))
    for seq in cluster:
        plt.plot(seq, alpha=0.4)
    plt.title(f'Cluster {i+1}')
    plt.xlabel('Mese')
    plt.ylabel('Numero di passeggeri')
    plt.show()
