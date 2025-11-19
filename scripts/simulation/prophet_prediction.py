#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTRAÎNEMENT MODÈLES PROPHET - NATIONAL + DÉPARTEMENTS
Compatible avec ton pipeline Ridge existant
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

print("="*80)
print(" ENTRAÎNEMENT PROPHET - NATIONAL + TOP DÉPARTEMENTS ".center(80, "="))
print("="*80)

# =============================================================================
# 1. CHARGEMENT DONNÉES (réutilise tes CSV)
# =============================================================================
print("\n[1/4] Chargement des données...")

df_nat = pd.read_csv('data/grippe-passages-aux-urgences-et-actes-sos-medecins-france.csv')
df_dept = pd.read_csv('data/grippe-passages-aux-urgences-et-actes-sos-medecins-departement.csv')
df_couv_nat = pd.read_csv('data/couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-france.csv')
df_couv_dept = pd.read_csv('data/couvertures-vaccinales-des-adolescent-et-adultes-departement.csv')

print(f"   ✓ National : {df_nat.shape[0]} semaines")
print(f"   ✓ Départemental : {df_dept.shape[0]} observations")

# =============================================================================
# 2. PRÉPARATION FORMAT PROPHET (ds + y + regressors)
# =============================================================================
print("\n[2/4] Préparation format Prophet...")

# --- NIVEAU NATIONAL ---
df_nat['ds'] = pd.to_datetime(df_nat['1er jour de la semaine'], errors='coerce')
df_nat_agg = df_nat.groupby('ds').agg({
    'Taux de passages aux urgences pour grippe': 'mean'
}).reset_index()
df_nat_agg.rename(columns={'Taux de passages aux urgences pour grippe': 'y'}, inplace=True)

# Merge avec couverture vaccinale (regressor)
df_couv_nat['Année'] = pd.to_datetime(df_couv_nat['Année'], format='%Y')
df_nat_agg['year'] = df_nat_agg['ds'].dt.year
df_nat_merged = df_nat_agg.merge(
    df_couv_nat[['Année', 'Grippe 65 ans et plus']],
    left_on=df_nat_agg['ds'].dt.year,
    right_on=df_couv_nat['Année'].dt.year,
    how='left'
)
df_nat_merged['vaccination_rate'] = df_nat_merged['Grippe 65 ans et plus'].fillna(method='ffill')
df_nat_prophet = df_nat_merged[['ds', 'y', 'vaccination_rate']].dropna()

print(f"   ✓ National Prophet : {df_nat_prophet.shape[0]} semaines")

# --- NIVEAU DÉPARTEMENTAL (top 10 départements par population) ---
top_depts = [
    'Paris', 'Nord', 'Bouches-du-Rhône', 'Rhône', 'Haute-Garonne',
    'Pas-de-Calais', 'Loire-Atlantique', 'Gironde', 'Seine-Saint-Denis', 'Yvelines'
]

df_dept['ds'] = pd.to_datetime(df_dept['1er jour de la semaine'], errors='coerce')
df_dept_prophet = {}

for dept in top_depts:
    df_d = df_dept[df_dept['Département'] == dept].copy()
    df_d_agg = df_d.groupby('ds').agg({
        'Taux de passages aux urgences pour grippe': 'mean'
    }).reset_index()
    df_d_agg.rename(columns={'Taux de passages aux urgences pour grippe': 'y'}, inplace=True)
    
    # Merge vaccination
    df_d_agg['year'] = df_d_agg['ds'].dt.year
    df_d_merged = df_d_agg.merge(
        df_couv_dept[df_couv_dept['Département'] == dept][['Année', 'Grippe 65 ans et plus']],
        left_on='year',
        right_on='Année',
        how='left'
    )
    df_d_merged['vaccination_rate'] = df_d_merged['Grippe 65 ans et plus'].fillna(method='ffill')
    df_dept_prophet[dept] = df_d_merged[['ds', 'y', 'vaccination_rate']].dropna()
    
    print(f"   ✓ {dept:20s} : {df_dept_prophet[dept].shape[0]} semaines")

# =============================================================================
# 3. ENTRAÎNEMENT PROPHET
# =============================================================================
print("\n[3/4] Entraînement des modèles Prophet...")

def train_prophet(df, name="National"):
    """Entraîne un modèle Prophet optimisé pour la grippe"""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # ⚠️ CRUCIAL pour les épidémies
        changepoint_prior_scale=0.05,       # Réduit l'overfitting
        interval_width=0.95,
        seasonality_prior_scale=10
    )
    
    # Ajoute vaccination comme regressor
    model.add_regressor('vaccination_rate', prior_scale=0.5)
    
    # Entraîne
    model.fit(df)
    
    # Métriques (train set)
    forecast = model.predict(df)
    mae = np.mean(np.abs(df['y'] - forecast['yhat']))
    
    print(f"   ✓ {name:20s} : MAE = {mae:.2f}")
    
    return model

# --- MODÈLE NATIONAL ---
model_national = train_prophet(df_nat_prophet, "National")

# --- MODÈLES DÉPARTEMENTAUX ---
models_dept = {}
for dept, df_d in df_dept_prophet.items():
    if len(df_d) >= 104:  # Au moins 2 ans de données
        models_dept[dept] = train_prophet(df_d, dept)

# =============================================================================
# 4. SAUVEGARDE
# =============================================================================
print("\n[4/4] Sauvegarde des modèles...")

Path('models/prophet').mkdir(parents=True, exist_ok=True)

# Sauvegarde national
joblib.dump(model_national, 'models/prophet/model_national.pkl')
joblib.dump(df_nat_prophet, 'models/prophet/data_national.pkl')

# Sauvegarde départements
for dept, model in models_dept.items():
    dept_safe = dept.replace(' ', '_').replace('-', '_')
    joblib.dump(model, f'models/prophet/model_{dept_safe}.pkl')
    joblib.dump(df_dept_prophet[dept], f'models/prophet/data_{dept_safe}.pkl')

print(f"\n   ✓ Modèle national sauvegardé")
print(f"   ✓ {len(models_dept)} modèles départementaux sauvegardés")

# Méta-données
metadata = {
    'date_entrainement': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
    'n_semaines_national': len(df_nat_prophet),
    'departements': list(models_dept.keys()),
    'periode': f"{df_nat_prophet['ds'].min().year}-{df_nat_prophet['ds'].max().year}",
    'regressor': 'vaccination_rate (Grippe 65+)'
}
joblib.dump(metadata, 'models/prophet/metadata.pkl')

print("\n" + "="*80)
print(" ✓ ENTRAÎNEMENT TERMINÉ ".center(80))
print("="*80)