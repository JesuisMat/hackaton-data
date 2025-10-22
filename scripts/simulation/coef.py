#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXTRACTION COEFFICIENTS AMÉLIORÉE - VERSION V2
Améliorations: Variables temporelles, démographiques, validation croisée
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import warnings
import sys
import io
warnings.filterwarnings('ignore')

print("="*80)
print(" EXTRACTION COEFFICIENTS V2 - MODÈLE ENRICHI ".center(80, "="))
print("="*80)

# Configuration encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
else:
    sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
print("\n [1/6] Chargement des datasets...")

df_nat = pd.read_csv('data/grippe-passages-aux-urgences-et-actes-sos-medecins-france.csv')
df_reg = pd.read_csv('data/grippe-passages-urgences-et-actes-sos-medecin_reg.csv')
df_dept = pd.read_csv('data/grippe-passages-aux-urgences-et-actes-sos-medecins-departement.csv')

df_couv_nat = pd.read_csv('data/couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-france.csv')
df_couv_dept = pd.read_csv('data/couvertures-vaccinales-des-adolescent-et-adultes-departement.csv')

print(f"   National : {df_nat.shape[0]} observations")
print(f"   Départemental : {df_dept.shape[0]} observations")
print(f"   Couvertures vaccinales : {df_couv_dept.shape[0]} observations")

# =============================================================================
# 2. PRÉPARATION AVEC VARIABLES ENRICHIES
# =============================================================================
print("\n [2/6] Enrichissement des variables...")

# Convertir dates
for df in [df_nat, df_reg, df_dept]:
    df['Date'] = pd.to_datetime(df['1er jour de la semaine'], errors='coerce')
    df['Annee'] = df['Date'].dt.year
    df['Mois'] = df['Date'].dt.month
    df['Semaine'] = df['Date'].dt.isocalendar().week
    df['Trimestre'] = df['Date'].dt.quarter

# Variables temporelles saisonnières
df_dept['Periode_pic'] = df_dept['Semaine'].apply(lambda x: 1 if (x >= 49 or x <= 10) else 0)
df_dept['Saison_automne'] = df_dept['Trimestre'].apply(lambda x: 1 if x == 4 else 0)
df_dept['Saison_hiver'] = df_dept['Trimestre'].apply(lambda x: 1 if x == 1 else 0)

# Agréger par département-année
df_dept_agg = df_dept.groupby(['Département', 'Annee']).agg({
    'Taux de passages aux urgences pour grippe': 'mean',
    'Taux d\'hospitalisations après passages aux urgences pour grippe': 'mean',
    'Taux d\'actes médicaux SOS médecins pour grippe': 'mean',
    'Région': 'first',
    'Periode_pic': 'mean',  # Proportion de semaines en période de pic
    'Saison_automne': 'mean',
    'Saison_hiver': 'mean'
}).reset_index()

# Fusion avec couvertures
df_merged = df_dept_agg.merge(
    df_couv_dept[['Département', 'Année', 'Grippe 65 ans et plus']],
    left_on=['Département', 'Annee'],
    right_on=['Département', 'Année'],
    how='inner'
)

df_merged.rename(columns={'Grippe 65 ans et plus': 'Couverture_65plus'}, inplace=True)

# Supprimer NaN
df_merged = df_merged.dropna(subset=[
    'Couverture_65plus',
    'Taux de passages aux urgences pour grippe',
    'Taux d\'actes médicaux SOS médecins pour grippe'
])

print(f"   Fusion réussie : {df_merged.shape[0]} observations")
print(f"   Départements avec données : {df_merged['Département'].nunique()}")

# =============================================================================
# 3. VARIABLES DÉRIVÉES
# =============================================================================
print("\n[3/6] Création variables dérivées...")

# Interaction couverture * période pic (effet non-linéaire)
df_merged['Couv_x_pic'] = df_merged['Couverture_65plus'] * df_merged['Periode_pic']

# Carré de la couverture (capturer non-linéarité)
df_merged['Couverture_sq'] = df_merged['Couverture_65plus'] ** 2

# Effet retardé (lag) de la couverture (approximation)
# Trier par département et année
df_merged = df_merged.sort_values(['Département', 'Annee'])
df_merged['Couverture_lag1'] = df_merged.groupby('Département')['Couverture_65plus'].shift(1)

# Remplir NaN du lag par valeur actuelle
df_merged['Couverture_lag1'] = df_merged['Couverture_lag1'].fillna(df_merged['Couverture_65plus'])

# Variables région (one-hot encoding)
region_dummies = pd.get_dummies(df_merged['Région'], prefix='Reg', drop_first=True)
df_merged = pd.concat([df_merged, region_dummies], axis=1)

print(f"   Variables créées : couv*pic, couv², couv_lag, régions dummies")

# =============================================================================
# 4. MODÈLE 1 : RÉGRESSION LINÉAIRE SIMPLE (BASELINE)
# =============================================================================
print("\n[4/6] Modèle 1 : Régression linéaire simple...")

# Variables de base
X_simple = df_merged[['Couverture_65plus', 'Taux d\'actes médicaux SOS médecins pour grippe']]
y = df_merged['Taux de passages aux urgences pour grippe']

# Split
X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# Standardisation
scaler_simple = StandardScaler()
X_train_scaled = scaler_simple.fit_transform(X_train)
X_test_scaled = scaler_simple.transform(X_test)

# Régression
model_simple = LinearRegression()
model_simple.fit(X_train_scaled, y_train)

# Scores
r2_simple = model_simple.score(X_test_scaled, y_test)
y_pred_simple = model_simple.predict(X_test_scaled)
rmse_simple = np.sqrt(np.mean((y_test - y_pred_simple)**2))
mae_simple = np.mean(np.abs(y_test - y_pred_simple))

print(f"\n   MODÈLE SIMPLE :")
print(f"      • R² : {r2_simple:.3f} ({r2_simple*100:.1f}%)")
print(f"      • RMSE : ±{rmse_simple:.2f}")
print(f"      • MAE : ±{mae_simple:.2f}")

# =============================================================================
# 5. MODÈLE 2 : RÉGRESSION ENRICHIE
# =============================================================================
print("\n[5/6] Modèle 2 : Régression enrichie (temporelle + non-linéaire)...")

# Variables enrichies
features_enrichies = [
    'Couverture_65plus',
    'Taux d\'actes médicaux SOS médecins pour grippe',
    'Periode_pic',
    'Couv_x_pic',
    'Couverture_sq',
    'Couverture_lag1'
]

# Ajouter dummies régions
region_cols = [col for col in df_merged.columns if col.startswith('Reg_')]
features_enrichies.extend(region_cols)

X_enrichi = df_merged[features_enrichies]

# Split
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_enrichi, y, test_size=0.2, random_state=42)

# Standardisation
scaler_enrichi = StandardScaler()
X_train_e_scaled = scaler_enrichi.fit_transform(X_train_e)
X_test_e_scaled = scaler_enrichi.transform(X_test_e)

# Ridge Regression (pénalisation pour éviter overfitting)
model_enrichi = Ridge(alpha=1.0)
model_enrichi.fit(X_train_e_scaled, y_train_e)

# Scores
r2_enrichi = model_enrichi.score(X_test_e_scaled, y_test_e)
y_pred_enrichi = model_enrichi.predict(X_test_e_scaled)
rmse_enrichi = np.sqrt(np.mean((y_test_e - y_pred_enrichi)**2))
mae_enrichi = np.mean(np.abs(y_test_e - y_pred_enrichi))

# Validation croisée
cv_scores = cross_val_score(model_enrichi, X_train_e_scaled, y_train_e, cv=5, scoring='r2')

print(f"\n   MODÈLE ENRICHI (Ridge) :")
print(f"      • R² : {r2_enrichi:.3f} ({r2_enrichi*100:.1f}%)")
print(f"      • RMSE : ±{rmse_enrichi:.2f}")
print(f"      • MAE : ±{mae_enrichi:.2f}")
print(f"      • CV R² moyen : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Importance des variables
importances = pd.DataFrame({
    'Variable': features_enrichies,
    'Coefficient': model_enrichi.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"\n TOP 5 VARIABLES IMPORTANTES :")
for idx, row in importances.head(5).iterrows():
    print(f"      • {row['Variable']:40s} : {row['Coefficient']:+.4f}")

# =============================================================================
# 6. COMPARAISON ET CHOIX DU MEILLEUR MODÈLE
# =============================================================================
print("\n [6/6] Comparaison et sélection...")

print(f"\n   COMPARAISON DES MODÈLES :")
print(f"   {'Modèle':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
print(f"   {'-'*50}")
print(f"   {'Simple (2 vars)':<20} {r2_simple:.3f}     {rmse_simple:.2f}      {mae_simple:.2f}")
print(f"   {'Enrichi (Ridge)':<20} {r2_enrichi:.3f}     {rmse_enrichi:.2f}      {mae_enrichi:.2f}")

# Amélioration relative
amelioration_r2 = ((r2_enrichi - r2_simple) / r2_simple) * 100
amelioration_rmse = ((rmse_simple - rmse_enrichi) / rmse_simple) * 100

print(f"\n   AMÉLIORATION MODÈLE ENRICHI :")
print(f"      • R² : +{amelioration_r2:.1f}%")
print(f"      • RMSE : -{amelioration_rmse:.1f}% d'erreur")

# Choisir le meilleur modèle
if r2_enrichi > r2_simple and rmse_enrichi < rmse_simple:
    print(f"\n   MODÈLE RETENU : Enrichi (Ridge)")
    model_final = model_enrichi
    scaler_final = scaler_enrichi
    r2_final = r2_enrichi
    rmse_final = rmse_enrichi
    mae_final = mae_enrichi
    features_final = features_enrichies
    
    # Extraire coefficients principaux
    idx_couv = features_enrichies.index('Couverture_65plus')
    idx_sos = features_enrichies.index('Taux d\'actes médicaux SOS médecins pour grippe')
    coef_couverture = model_enrichi.coef_[idx_couv]
    coef_sos = model_enrichi.coef_[idx_sos]
else:
    print(f"\n   MODÈLE RETENU : Simple (plus stable)")
    model_final = model_simple
    scaler_final = scaler_simple
    r2_final = r2_simple
    rmse_final = rmse_simple
    mae_final = mae_simple
    features_final = ['Couverture_65plus', 'Taux d\'actes médicaux SOS médecins pour grippe']
    coef_couverture = model_simple.coef_[0]
    coef_sos = model_simple.coef_[1]

intercept = model_final.intercept_

# =============================================================================
# 7. SAUVEGARDE COEFFICIENTS
# =============================================================================
print("\n [7/7] Sauvegarde des coefficients...")

simulation_params = {
    'coefficients': {
        'vaccination_urgences': coef_couverture,
        'sos_urgences': coef_sos,
        'intercept': intercept
    },
    'model_performance': {
        'r2': r2_final,
        'rmse': rmse_final,
        'mae': mae_final,
        'model_type': 'Ridge' if model_final == model_enrichi else 'LinearRegression'
    },
    'scaler': {
        'mean': scaler_final.mean_.tolist(),
        'std': scaler_final.scale_.tolist()
    },
    'features': features_final,
    'elasticites': {
        'vaccination': {
            'description': 'Impact +1 point couverture → Δ urgences',
            'valeur': coef_couverture,
            'unite': 'passages/100k par point de couverture'
        },
        'sos_medecins': {
            'description': 'Impact +1 acte SOS/100k → Δ urgences',
            'valeur': coef_sos,
            'unite': 'passages/100k par acte SOS'
        }
    },
    'metadata': {
        'n_observations': len(df_merged),
        'n_departements': df_merged['Département'].nunique(),
        'periode': f"{df_merged['Annee'].min()}-{df_merged['Annee'].max()}",
        'variables_enrichies': len(features_final)
    }
}

# Élasticité %
mean_couv = df_merged['Couverture_65plus'].mean()
mean_urg = df_merged['Taux de passages aux urgences pour grippe'].mean()
elasticite_pct = (coef_couverture * mean_couv) / mean_urg * 100

simulation_params['elasticites']['vaccination_pct'] = {
    'description': 'Impact +1% couverture → Δ% urgences',
    'valeur': elasticite_pct,
    'unite': '% variation'
}

# Sauvegarder
joblib.dump(simulation_params, 'models/simulation_coefficients_v2.pkl')
joblib.dump(model_final, 'models/regression_model_v2.pkl')
joblib.dump(scaler_final, 'models/scaler_v2.pkl')

print(f"\n   Fichiers sauvegardés :")
print(f"      • simulation_coefficients_v2.pkl")
print(f"      • regression_model_v2.pkl")
print(f"      • scaler_v2.pkl")

# =============================================================================
# 8. ANALYSE COMPLÉMENTAIRE
# =============================================================================
print("\n" + "="*80)
print(" INSIGHTS COMPLÉMENTAIRES ".center(80, "="))
print("="*80)

# Corrélation
corr_couv_urg = df_merged['Couverture_65plus'].corr(
    df_merged['Taux de passages aux urgences pour grippe']
)

print(f"\n CORRÉLATION SIMPLE :")
print(f"   • Couverture ↔ Urgences : r = {corr_couv_urg:.3f}")

if corr_couv_urg < 0:
    print(f" Corrélation NÉGATIVE : + vaccination → - urgences (attendu)")
else:
    print(f"  Corrélation POSITIVE : À investiguer (causalité inverse?)")

# Effet saisonnier
effet_pic = df_merged.groupby('Periode_pic')['Taux de passages aux urgences pour grippe'].mean()

# Vérifier que les données existent
if len(effet_pic) >= 2:
    hors_pic = effet_pic.iloc[0]  # Premier élément
    en_pic = effet_pic.iloc[1]    # Deuxième élément
    print(f"   • Hors pic (printemps-été) : {hors_pic:.1f} urgences/100k")
    print(f"   • Période pic (hiver) : {en_pic:.1f} urgences/100k")
    print(f"   • Ratio pic/hors-pic : {en_pic/hors_pic:.2f}x")
else:
    print(f"   ⚠️  Données de saisonnalité insuffisantes")

# Top/Bottom départements
df_recent = df_merged[df_merged['Annee'] == df_merged['Annee'].max()]
top_couv = df_recent.nlargest(5, 'Couverture_65plus')[
    ['Département', 'Couverture_65plus', 'Taux de passages aux urgences pour grippe']
]
bottom_couv = df_recent.nsmallest(5, 'Couverture_65plus')[
    ['Département', 'Couverture_65plus', 'Taux de passages aux urgences pour grippe']
]

print(f"\n TOP 5 DÉPARTEMENTS (meilleure couverture {int(df_merged['Annee'].max())}) :")
for idx, row in top_couv.iterrows():
    print(f"   • {row['Département']:15s} : {row['Couverture_65plus']:.1f}% → {row['Taux de passages aux urgences pour grippe']:.1f} urg/100k")

print(f"\n  BOTTOM 5 DÉPARTEMENTS (pire couverture {int(df_merged['Annee'].max())}) :")
for idx, row in bottom_couv.iterrows():
    print(f"   • {row['Département']:15s} : {row['Couverture_65plus']:.1f}% → {row['Taux de passages aux urgences pour grippe']:.1f} urg/100k")

# Potentiel
gap_couv = top_couv['Couverture_65plus'].mean() - bottom_couv['Couverture_65plus'].mean()
gap_urg = bottom_couv['Taux de passages aux urgences pour grippe'].mean() - top_couv['Taux de passages aux urgences pour grippe'].mean()

print(f"\n POTENTIEL D'AMÉLIORATION :")
print(f"   • Écart couverture top/bottom : {gap_couv:.1f} points")
print(f"   • Écart urgences bottom/top : {gap_urg:.1f} passages/100k")
print(f"   • Si bottom atteignait niveau top : -{gap_urg:.1f} urgences/100k ({-gap_urg/bottom_couv['Taux de passages aux urgences pour grippe'].mean()*100:.1f}%)")

print("\n" + "="*80)
print(" EXTRACTION COEFFICIENTS V2 TERMINÉE ".center(80))
print("="*80)
print(f"\n   Modèle prêt avec {len(features_final)} variables!\n")