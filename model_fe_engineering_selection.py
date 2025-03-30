import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Cargar datos
df = pd.read_csv("SeoulBikeData.csv", encoding="ISO-8859-1")

# Preprocesamiento de datos
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month
df["Is_Weekend"] = df["Date"].dt.dayofweek >= 5  # 5 y 6 son sábado y domingo
df.drop(columns=["Date"], inplace=True)

# Definir variables categóricas y numéricas
categorical_features = [
    "Seasons", "Holiday", "Functioning Day", "Hour", "Month", "Is_Weekend"
]
numeric_features = [
    "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)", "Visibility (10m)",
    "Dew point temperature(°C)", "Solar Radiation (MJ/m2)", "Rainfall(mm)",
    "Snowfall (cm)"
]

# Separar datos
X = df[categorical_features + numeric_features]
y = df["Rented Bike Count"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Construir transformadores
preprocessor = ColumnTransformer(transformers=[
    ("num", PowerTransformer(method="yeo-johnson"), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Aplicar selección de características
selector = VarianceThreshold(threshold=0.01)

# Evaluar modelos con distintos valores de k
best_k, best_rmse, best_model = None, float("inf"), None
k_values = [3, 5, 10, 15, 20, 50, 100, 300, 500, 1000]

for k in k_values:
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("knn", KNeighborsRegressor(n_neighbors=k))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    if rmse < best_rmse:
        best_k, best_rmse, best_model = k, rmse, model

# Guardar el mejor modelo
with open("model_fe_engineering_selection.pk", "wb") as f:
    pickle.dump(best_model, f)

print(f"Mejor modelo: KNN con k={best_k}, RMSE={best_rmse:.2f}")
