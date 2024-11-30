import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from lib.ts_manip import sliding_win_cluster_aware
import numpy as np
from sklearn.metrics import silhouette_score

# Genera dati di esempio
path = "./datasets/processed/019_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z1_5000_6168_6212.txt" # For relative paths
    
train_perc = 0.7
series = np.loadtxt(path)
train_size = int(len(series) * train_perc)
# Rimuovi componenti anomale.
train_series = series[0:train_size]
test_series = series[train_size:]

plot = False
if plot: 
    # plotta la Serie temporale.
    plt.plot(series)
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.show()
    #exit(1)
window_size_cluster = 75
window_size_tree = 50
win_pred = 1   
n_clusters = 5 # Dimezzati perchè abbiamo train e test    
train_wins_cluster, train_wins_tree, train_target_tree = sliding_win_cluster_aware(series = train_series, window_size_cluster = window_size_cluster, window_size_pred = window_size_tree, win_out_pred = win_pred)
print(f"La dimensione del cluster training è {len(train_wins_cluster)}")
print(f"La dimensione del tree training è {len(train_target_tree)}")

test_wins_cluster, test_wins_tree, test_target_tree = sliding_win_cluster_aware(series = test_series, window_size_cluster = window_size_cluster, window_size_pred = window_size_tree, win_out_pred = win_pred)

print(f"La dimensione del cluster testing è {len(train_wins_cluster)}")
print(f"La dimensione del tree testing è {len(test_wins_cluster)}")
print(f"Individuiamo il numero di cluster ottimo.")

inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(train_wins_cluster)
    inertia.append(kmeans.inertia_)

# Traccia il grafico del gomito
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Numero di cluster k')
plt.ylabel('Inertia')
plt.title('Metodo del Gomito per determinare k ottimale')
plt.show()

silhouette_scores = []
K = range(2, 11)  # Silhouette non definito per k=1
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(train_wins_cluster)
    score = silhouette_score(train_wins_cluster, labels)
    silhouette_scores.append(score)

# Traccia il grafico dei punteggi di silhouette
plt.figure(figsize=(8, 4))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Numero di cluster k')
plt.ylabel('Punteggio di Silhouette')
plt.title('Analisi del Silhouette per determinare k ottimale')
plt.show()