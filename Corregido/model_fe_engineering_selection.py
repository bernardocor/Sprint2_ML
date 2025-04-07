
"""
Recomendación: utilizar un linter como 'pycodestyle' o 'black' 
para validar el cumplimiento del estándar PEP8 antes de entregas.
Instalación sugerida: pip install pycodestyle black
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Cargar datos
df = pd.read_csv("SeoulBikeData.csv", encoding="ISO-8859-1")

# Preprocesamiento de datos
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df.sort_values(by=["Date", "Hour"], inplace=True)
df["Month"] = df["Date"].dt.month
df["Is_Weekend"] = df["Date"].dt.dayofweek >= 5
df.drop(columns=["Date"], inplace=True)

# Definir variables categóricas y numéricas
categorical_features = [
    "Seasons", "Holiday", "Functioning Day", "Hour", "Month", "Is_Weekend"
]
numeric_features = [
    "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)",
    "Visibility (10m)", "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)", "Rainfall(mm)", "Snowfall (cm)"
]

# Separar datos en entrenamiento y prueba de forma temporal
train_size = int(len(df) * 0.8)
X_train = df.iloc[:train_size][categorical_features + numeric_features]
y_train = df.iloc[:train_size]["Rented Bike Count"]
X_test = df.iloc[train_size:][categorical_features + numeric_features]
y_test = df.iloc[train_size:]["Rented Bike Count"]

# Construir transformadores (pipeline)
preprocessor = ColumnTransformer(transformers=[
    ("num", PowerTransformer(method="yeo-johnson"), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Evaluar modelos KNN para diferentes valores de k
best_k, best_rmse, best_model = None, float("inf"), None
k_values = [3, 5, 10, 15, 20, 50, 100, 300, 500, 1000]

for k in k_values:
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("knn", KNeighborsRegressor(n_neighbors=k))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    if rmse < best_rmse:
        best_k, best_rmse, best_model = k, rmse, model

# Guardar el mejor modelo entrenado
with open("model_fe_engineering_selection.pk", "wb") as f:
    pickle.dump(best_model, f)

# Mostrar resultados
print(f"Mejor modelo: KNN con k={best_k}, RMSE={best_rmse:.2f}")
