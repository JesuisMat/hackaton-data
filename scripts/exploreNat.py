#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*80)
print(" ANALYSE NATIONALE - GRIPPE FRANCE ".center(80, "="))
print("="*80)
print(f"🕐 Début : {datetime.now().strftime('%H:%M:%S')}\n")

# =============================================================================
# CHEMINS
# =============================================================================
# Détection du répertoire racine du projet
if os.path.exists('../data'):  # Exécuté depuis scripts/
    DATA_DIR = '../data'
    OUTPUT_DIR = '../output'
elif os.path.exists('./data'):  # Exécuté depuis racine
    DATA_DIR = './data'
    OUTPUT_DIR = './output'
else:
    raise FileNotFoundError("Impossible de trouver le dossier 'data'. Vérifiez votre structure.")

# Création du dossier output s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE = f"{DATA_DIR}/grippe-passages-aux-urgences-et-actes-sos-medecins-france.csv"

print(f"📁 Chemins configurés :")
print(f"   • Données : {INPUT_FILE}")
print(f"   • Outputs : {OUTPUT_DIR}/\n")

# =============================================================================
# 1. CHARGEMENT
# =============================================================================
print("📂 [1/5] Chargement des données...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Fichier introuvable : {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
print(f"   ✓ {df.shape[0]} lignes × {df.shape[1]} colonnes chargées")

# =============================================================================
# 2. NETTOYAGE
# =============================================================================
print("\n🧹 [2/5] Nettoyage et préparation...")

# Conversion de la colonne date avec gestion des erreurs
df['Date'] = pd.to_datetime(df['1er jour de la semaine'], errors='coerce')

# Extraction des composantes temporelles
df['Annee'] = df['Date'].dt.year
df['Mois'] = df['Date'].dt.month
df['Semaine'] = df['Date'].dt.isocalendar().week

# Tri par date et réinitialisation de l'index
df = df.sort_values('Date').reset_index(drop=True)

# Calcul du pourcentage de complétude des données
completude = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100

# Affichage des informations
print(f"   ✓ Période : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"   ✓ Durée : {(df['Date'].max() - df['Date'].min()).days} jours")
print(f"   ✓ Classes d'âge : {df['Classe d\'âge'].nunique()}")  # Notez l'échappement du guillemet
print(f"   ✓ Complétude : {completude:.1f}%")

# =============================================================================
# 3. STATISTIQUES
# =============================================================================
print("\n📊 [3/5] Calcul des statistiques...")

# Moyennes par classe d'âge
stats = df.groupby('Classe d\'âge').agg({
    'Taux de passages aux urgences pour grippe': 'mean',
    'Taux d\'hospitalisations après passages aux urgences pour grippe': 'mean',
    'Taux d\'actes médicaux SOS médecins pour grippe': 'mean'
}).round(2)

stats_sorted = stats.sort_values('Taux de passages aux urgences pour grippe', ascending=False)

print(f"\n   📋 TOP 3 classes d'âge (passages urgences) :")
for i, (classe, row) in enumerate(stats_sorted.head(3).iterrows(), 1):
    print(f"      {i}. {classe:20s} : {row['Taux de passages aux urgences pour grippe']:6.1f} pour 100k")

# Corrélation Urgences / SOS Médecins
corr = df['Taux de passages aux urgences pour grippe'].corr(
    df['Taux d\'actes médicaux SOS médecins pour grippe']
)
print(f"\n   🔗 Corrélation Urgences ↔ SOS Médecins : r = {corr:.3f}")

# Mois le plus critique
mois_critique = df.groupby('Mois')['Taux de passages aux urgences pour grippe'].mean().idxmax()
taux_critique = df.groupby('Mois')['Taux de passages aux urgences pour grippe'].mean().max()
mois_noms = {1:'Jan', 2:'Fév', 3:'Mar', 4:'Avr', 5:'Mai', 6:'Jun',
             7:'Jul', 8:'Aoû', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Déc'}
print(f"   📅 Mois le + critique : {mois_noms[mois_critique]} ({taux_critique:.1f} pour 100k)")

# =============================================================================
# 4. VISUALISATIONS
# =============================================================================
print("\n📈 [4/5] Génération des graphiques...")

# --- GRAPHIQUE 1 : Évolution temporelle ---
print("   → Graphique 1/3 : Évolution temporelle...")

fig, axes = plt.subplots(3, 1, figsize=(16, 11))
fig.suptitle('📈 ÉVOLUTION NATIONALE DES INDICATEURS GRIPPE', 
             fontsize=16, fontweight='bold', y=0.995)

classes = sorted(df['Classe d\'âge'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

for idx, classe in enumerate(classes):
    data = df[df['Classe d\'âge'] == classe]
    
    axes[0].plot(data['Date'], data['Taux de passages aux urgences pour grippe'],
                 label=classe, linewidth=2, color=colors[idx], alpha=0.85)
    
    axes[1].plot(data['Date'], data['Taux d\'hospitalisations après passages aux urgences pour grippe'],
                 label=classe, linewidth=2, color=colors[idx], alpha=0.85)
    
    axes[2].plot(data['Date'], data['Taux d\'actes médicaux SOS médecins pour grippe'],
                 label=classe, linewidth=2, color=colors[idx], alpha=0.85)

titres = [
    '🚨 Taux de passages aux urgences (pour 100k hab)',
    '🏥 Taux d\'hospitalisation après urgences (%)',
    '🚑 Taux d\'actes SOS Médecins (pour 100k hab)'
]

for ax, titre in zip(axes, titres):
    ax.set_title(titre, fontsize=11, fontweight='bold', pad=8)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_xlabel('Date', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_evolution_nationale.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ 01_evolution_nationale.png")

# --- GRAPHIQUE 2 : Saisonnalité ---
print("   → Graphique 2/3 : Saisonnalité...")

pivot = df.pivot_table(
    values='Taux de passages aux urgences pour grippe',
    index='Classe d\'âge',
    columns='Mois',
    aggfunc='mean'
)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': 'Taux moyen (pour 100k)'}, linewidths=0.5, ax=ax)

ax.set_title('📅 SAISONNALITÉ : Taux moyen de passages urgences par mois', 
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Mois', fontsize=11)
ax.set_ylabel('Classe d\'âge', fontsize=11)

# Renommer les mois
ax.set_xticklabels([mois_noms.get(int(col), col) for col in pivot.columns], rotation=0)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_saisonnalite.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ 02_saisonnalite.png")

# --- GRAPHIQUE 3 : Comparaison classes d'âge ---
print("   → Graphique 3/3 : Comparaison classes d'âge...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('📊 DISTRIBUTION PAR CLASSE D\'ÂGE', fontsize=14, fontweight='bold')

# Boxplot 1
sns.boxplot(data=df, x='Classe d\'âge', y='Taux de passages aux urgences pour grippe',
            palette='Set2', ax=axes[0])
axes[0].set_title('🚨 Passages urgences', fontweight='bold', fontsize=11)
axes[0].set_xlabel('')
axes[0].set_ylabel('Taux (pour 100k)', fontsize=10)
axes[0].tick_params(axis='x', rotation=45, labelsize=9)

# Boxplot 2
sns.boxplot(data=df, x='Classe d\'âge', y='Taux d\'hospitalisations après passages aux urgences pour grippe',
            palette='Set2', ax=axes[1])
axes[1].set_title('🏥 Hospitalisations', fontweight='bold', fontsize=11)
axes[1].set_xlabel('')
axes[1].set_ylabel('Taux (%)', fontsize=10)
axes[1].tick_params(axis='x', rotation=45, labelsize=9)

# Boxplot 3
sns.boxplot(data=df, x='Classe d\'âge', y='Taux d\'actes médicaux SOS médecins pour grippe',
            palette='Set2', ax=axes[2])
axes[2].set_title('🚑 SOS Médecins', fontweight='bold', fontsize=11)
axes[2].set_xlabel('')
axes[2].set_ylabel('Taux (pour 100k)', fontsize=10)
axes[2].tick_params(axis='x', rotation=45, labelsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_comparaison_classes_age.png', dpi=300, bbox_inches='tight')
plt.close()
print("      ✓ 03_comparaison_classes_age.png")

# =============================================================================
# 5. INSIGHTS
# =============================================================================
print("\n💡 [5/5] Insights clés\n")
print("="*80)
print(" SYNTHÈSE NATIONALE ".center(80, "="))
print("="*80)

print(f"\n🎯 CLASSE D'ÂGE LA + TOUCHÉE : {stats_sorted.index[0]}")
print(f"   → {stats_sorted.iloc[0]['Taux de passages aux urgences pour grippe']:.1f} passages/100k en moyenne")

print(f"\n📅 MOIS CRITIQUE : {mois_noms[mois_critique]}")
print(f"   → {taux_critique:.1f} passages/100k en moyenne")

print(f"\n🔗 CORRÉLATION URGENCES ↔ SOS MÉDECINS : {corr:.3f}")
if corr > 0.7:
    print("   → Forte corrélation : les deux indicateurs évoluent ensemble")
elif corr > 0.4:
    print("   → Corrélation modérée")
else:
    print("   → Faible corrélation : possibles disparités d'accès")

# Évolution annuelle
print(f"\n📈 ÉVOLUTION ANNUELLE (taux moyen urgences) :")
tendance = df.groupby('Annee')['Taux de passages aux urgences pour grippe'].mean().sort_index()
for annee, taux in tendance.items():
    evolution = ""
    if annee > tendance.index.min():
        diff = taux - tendance[annee-1]
        evolution = f" ({'+'if diff>0 else ''}{diff:.1f})"
    print(f"   • {int(annee)} : {taux:.1f} pour 100k{evolution}")

print("\n" + "="*80)
print("✅ ANALYSE NATIONALE TERMINÉE".center(80))
print("="*80)
print(f"🕐 Fin : {datetime.now().strftime('%H:%M:%S')}")
print(f"\n📊 3 graphiques générés dans {OUTPUT_DIR}/\n")