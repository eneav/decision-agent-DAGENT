from pathlib import Path       #für den pfad zur db
import sqlite3 #lokale db
import pandas as pd #data manipulation
import joblib #speicherunng des modells 
from sklearn.preprocessing import OneHotEncoder #katergorische variablen encodieren

#verbidnugn zur db herstellen. frei enschiedbar ob lokal oder cloud db | diese liegt lokal, läuft über dbeaver (skalebar)
DB_PATH = Path("data/personal.db")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()   





query = """SELECT alter_jahre, wohnlage, entfernung_km, homeoffice_tage, beziehungsstatus, kinder, krankheitstage FROM mitarbeiter"""   #DATEN WERDEN AUSGELESEN DATEN WERDEN AUSGELESEN DATEN WERDEN AUSGELESEN 
df = pd.read_sql_query(query, conn) #sql abfrage wird in ein dataframe umgewandelt

#one hot encoding für die kategorischen variablen | also encoder training 
from itertools import product  

wohnlage_vals = ["gut", "neutral", "schlecht"]
beziehungsstatus_vals = ["ledig", "verheiratet", "geschieden", "verwitwet"]
kombis = list(product(wohnlage_vals, beziehungsstatus_vals))   #alle kombinationen der kategorischen variablen
df_kombis = pd.DataFrame(kombis, columns=["wohnlage", "beziehungsstatus"])

encoder = OneHotEncoder(drop="first", sparse_output=False)
encoder.fit(df_kombis)





#vorbereitung der daten für das modell | also die features und labels
X = df.drop(columns=["krankheitstage"])
y = df["krankheitstage"]


categorical_cols = ["wohnlage", "beziehungsstatus"]
ohe_features = encoder.transform(X[categorical_cols])
ohe_df = pd.DataFrame(ohe_features, columns=encoder.get_feature_names_out(categorical_cols))
X_encoded = pd.concat([X.drop(columns=categorical_cols).reset_index(drop=True), ohe_df], axis=1)

# RFR modell erstellen und trainieren 


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor   #RFR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np #für die berechnung der fehler

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42) #train/test split
model = RandomForestRegressor(random_state=42) #modell initialisieren 
model.fit(X_train, y_train) #nmodel trainieren

#evaluieren des modells
y_pred = model.predict(X_test)
print("\nEvaluation:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")




MODEL_PATH = Path("../models/rf_regressor.pkl")   #speicherpfad 
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)  
joblib.dump(model, MODEL_PATH)
print(f"\n Modell gespeichert unter: {MODEL_PATH.resolve()}")
