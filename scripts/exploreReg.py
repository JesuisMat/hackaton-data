#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALYSE RÉGIONALE - DISPARITÉS TERRITORIALES
Chemins : data/grippe-passages-urgences-et-actes-sos-medecin_reg.csv → output/
Durée : ~8 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

print("="*80)
print(" ANALYSE RÉGIONALE - DISPARITÉS TERRITORIALES ".center(80, "="))
print("="*80)
print(f"🕐 Début : {datetime.now().strftime('%H:%M:%S')}\n")

# =============================================================================
# CHEMINS
# =============================================================================
if os.path.exists('../data'):
    DATA_DIR = '../data'
    OUTPUT_DIR = '../output'
elif os.path.exists('./data'):
    DATA_DIR = './data'
    OUTPUT_DIR = './output'
else:
    raise FileNotFoundError("Impossible de trouver le dossier 'data'")

os.makedirs(OUTPUT_DIR, exist_ok=True)
INPUT_FILE = f"{DATA_DIR}/grippe-passages-urgences-et-actes-sos-medecin_reg.csv"

print(f"📁 Données : {INPUT_FILE}")
print(f"📁 Outputs : {OUTPUT_DIR}/\n")

# =============================================================================
# 1. CHARGEMENT
# =============================================================================
print("📂 [1/5] Chargement des données...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Fichier introuvable : {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
print(f"   ✓ {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"   ✓ {df['Région'].nunique()} régions")

# =============================================================================
# 2. PRÉPARATION
# =============================================================================
print("\n🧹 [2/5] Préparation des données...")

df['Date'] = pd.to_datetime(df['1er jour de la semaine'], errors='coerce')
df['Annee'] = df['Date'].dt.year
df['Mois'] = df['Date'].dt.month
df = df.sort_values('Date')

print(f"   ✓ Période : {df['Date'].min().date()} → {df['Date'].max().date()}")

# =============================================================================
# 3. CALCUL INDICATEURS
# =============================================================================
print("\n📊 [3/5] Calcul des indicateurs...")

# Moyennes par région
taux_region = df.groupby('Région').agg({
    'Taux de passages aux urgences pour grippe': 'mean',
    'Taux d\'hospitalisations après passages aux urgences pour grippe': 'mean',
    'Taux d\'actes médicaux SOS médecins pour grippe': 'mean'
}).round(2)

taux_region = taux_region.sort_values('Taux de passages aux urgences pour grippe', ascending=False)

print(f"\n   🏆 TOP 5 RÉGIONS (urgences) :")
for i, (region, row) in enumerate(taux_region.head(5).iterrows(), 1):
    print(f"      {i}. {region:35s} : {row['Taux de passages aux urgences pour grippe']:6.1f} pour 100k")

print(f"\n   🟢 BOTTOM 5 RÉGIONS (urgences) :")
for i, (region, row) in enumerate(taux_region.tail(5).iterrows(), 1):
    print(f"      {i}. {region:35s} : {row['Taux de passages aux urgences pour grippe']:6.1f} pour 100k")

# Disparités
ratio = taux_region['Taux de passages aux urgences pour grippe'].max() / taux_region['Taux de passages aux urgences pour grippe'].min()
ecart = taux_region['Taux de passages aux urgences pour grippe'].max() - taux_region['Taux de passages aux urgences pour grippe'].min()

print(f"\n   ⚖️ DISPARITÉS :")
print(f"      • Écart max-min : {ecart:.1f} passages/100k")
print(f"      • Ratio max/min : {ratio:.2f}x")

# =============================================================================
# 4. VISUALISATIONS
# =============================================================================
print("\n📈 [4/5] Génération des graphiques...")

# --- GRAPHIQUE 1 : Classement régions ---
print("   → Graphique 1/3 : Classement des régions...")

fig, ax = plt.subplots(figsize=(12, 10))

regions = taux_region.index
valeurs = taux_region['Taux de passages aux urgences pour grippe'].values
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(regions)))

y_pos = np.arange(len(regions))
bars = ax.barh(y_pos, valeurs, color=colors, edgecolor='black', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(regions, fontsize=9)
ax.set_xlabel('Taux moyen passages urgences (pour 100k)', fontsize=11, fontweight='bold')
ax.set_title('🗺️ CLASSEMENT DES RÉGIONS - Pression sur les urgences', 
             fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Moyenne nationale
moyenne = valeurs.mean()
ax.axvline(moyenne, color='red', linestyle='--', linewidth=2.5, 
           label=f'Moyenne nationale ({moyenne:.1f})', alpha=0.8)
ax.legend(fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_classement_regions.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ 04_classement_regions.png")

# --- GRAPHIQUE 2 : Évolution top vs bottom ---
print("   → Graphique 2/3 : Évolution temporelle...")

top5 = taux_region.head(5).index
bottom5 = taux_region.tail(5).index

fig, ax = plt.subplots(figsize=(16, 8))

for region in top5:
    data = df[df['Région'] == region].groupby('Date')['Taux de passages aux urgences pour grippe'].mean()
    ax.plot(data.index, data.values, label=f'🔴 {region}', linewidth=2.5, alpha=0.85)

for region in bottom5:
    data = df[df['Région'] == region].groupby('Date')['Taux de passages aux urgences pour grippe'].mean()
    ax.plot(data.index, data.values, label=f'🟢 {region}', linewidth=1.8, 
            linestyle='--', alpha=0.65)

ax.set_title('📈 ÉVOLUTION TEMPORELLE : Top 5 vs Bottom 5', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Taux passages urgences (pour 100k)', fontsize=11)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_evolution_regions.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ 05_evolution_regions.png")

# --- GRAPHIQUE 3 : Hospitalisations ---
print("   → Graphique 3/3 : Taux d'hospitalisation...")

hospit = df.groupby('Région')['Taux d\'hospitalisations après passages aux urgences pour grippe'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 10))

y_pos = np.arange(len(hospit))
colors_hosp = plt.cm.Reds(np.linspace(0.3, 0.9, len(hospit)))
bars = ax.barh(y_pos, hospit.values, color=colors_hosp, edgecolor='black', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(hospit.index, fontsize=9)
ax.set_xlabel('Taux d\'hospitalisation moyen (%)', fontsize=11, fontweight='bold')
ax.set_title('🏥 TAUX D\'HOSPITALISATION APRÈS URGENCES - Par région', 
             fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Valeurs sur barres
for i, val in enumerate(hospit.values):
    ax.text(val + 0.02, i, f'{val:.2f}%', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_hospitalisations_regions.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ 06_hospitalisations_regions.png")

# =============================================================================
# 5. INSIGHTS
# =============================================================================
print("\n💡 [5/5] Insights clés\n")
print("="*80)
print(" SYNTHÈSE RÉGIONALE ".center(80, "="))
print("="*80)

print(f"\n🎯 TOP 3 RÉGIONS PRIORITAIRES :")
for i, region in enumerate(taux_region.head(3).index, 1):
    t_urg = taux_region.loc[region, 'Taux de passages aux urgences pour grippe']
    t_hosp = taux_region.loc[region, 'Taux d\'hospitalisations après passages aux urgences pour grippe']
    print(f"   {i}. {region}")
    print(f"      → Urgences : {t_urg:.1f} pour 100k | Hospitalisation : {t_hosp:.2f}%")

print(f"\n⚖️ INÉGALITÉS TERRITORIALES :")
print(f"   • Ratio max/min : {ratio:.2f}x")
print(f"   • La région la + touchée a un taux {ratio:.1f} fois supérieur à la - touchée")

au_dessus = (taux_region['Taux de passages aux urgences pour grippe'] > moyenne).sum()
print(f"\n📊 RÉPARTITION :")
print(f"   • {au_dessus}/{len(taux_region)} régions au-dessus de la moyenne nationale")
print(f"   • {len(taux_region) - au_dessus}/{len(taux_region)} régions en-dessous")

print("\n" + "="*80)
print("✅ ANALYSE RÉGIONALE TERMINÉE".center(80))
print("="*80)
print(f"🕐 Fin : {datetime.now().strftime('%H:%M:%S')}")
print(f"\n📊 3 graphiques générés dans {OUTPUT_DIR}/\n")