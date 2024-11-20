import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# Creazione di un dataset di esempio (serie temporale simulata)
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Definizione del modello
model = RandomForestRegressor(random_state=42)

# Definizione della grid search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Creazione del TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# GridSearchCV con TimeSeriesSplit
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',  # Usando MSE come metrica
    n_jobs=-1,
    verbose=1
)

# Esecuzione della grid search
grid_search.fit(X, y)

# Migliori parametri trovati
print("Migliori parametri:", grid_search.best_params_)

# Miglior punteggio
print("Miglior punteggio (MSE):", -grid_search.best_score_)