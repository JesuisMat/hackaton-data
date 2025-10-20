#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALYSE DÉPARTEMENTALE - ZONES MICRO-LOCALES
Chemins : data/grippepassagesauxurgencesetactessosmedecinsdepartement.csv → output/
Durée : ~10 minutes
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
print(" ANALYSE DÉPARTEMENTALE - ZONES CRITIQUES ".center(80, "="))
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
INPUT_FILE = f"{DATA_DIR}/grippe-passages-aux-urgences-et-actes-sos-medecins-departement.csv"

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
print(f"   ✓ {df['Département'].nunique()} départements")

# =============================================================================
# 2. PRÉPARATION
# =============================================================================
print("\n🧹 [2/5] Préparation des données...")

df['Date'] = pd.to_datetime(df['1er jour de la semaine'], errors='coerce')
df['Annee'] = df['Date'].dt.year
df['Mois'] = df['Date'].dt.month
df = df.sort_values('Date')

print(f"   ✓ Période : {df['Date'].min().date()} → {df['Date'].max().date()}")

# Qualité
qualite = df.groupby('Département')['Taux de passages aux urgences pour grippe'].apply(
    lambda x: 100 - (x.isna().sum() / len(x) * 100)
)
print(f"   ✓ Départements exploitables (>80%) : {(qualite > 80).sum()}/{len(qualite)}")

# =============================================================================
# 3. CALCUL INDICATEURS
# =============================================================================
print("\n📊 [3/5] Calcul des indicateurs...")

# Moyennes par département
taux_dept = df.groupby('Département').agg({
    'Taux de passages aux urgences pour grippe': 'mean',
    'Taux d\'hospitalisations après passages aux urgences pour grippe': 'mean',
    'Région': 'first'
}).sort_values('Taux de passages aux urgences pour grippe', ascending=False)

# Variabilité
variabilite = df.groupby('Département')['Taux de passages aux urgences pour grippe'].std().sort_values(ascending=False)

print(f"\n   🏆 TOP 10 DÉPARTEMENTS (urgences) :")
for i, (dept, row) in enumerate(taux_dept.head(10).iterrows(), 1):
    print(f"      {i:2d}. {dept:30s} ({row['Région']:22s}) : {row['Taux de passages aux urgences pour grippe']:6.1f} pour 100k")

# Départements à double risque
seuil_urg = taux_dept['Taux de passages aux urgences pour grippe'].quantile(0.75)
seuil_hosp = taux_dept['Taux d\'hospitalisations après passages aux urgences pour grippe'].quantile(0.75)

# Après avoir calculé les seuils (ligne ~100)
double_risque = taux_dept[
    (taux_dept['Taux de passages aux urgences pour grippe'] >= seuil_urg) &
    (taux_dept['Taux d\'hospitalisations après passages aux urgences pour grippe'] >= seuil_hosp)
]
print(f"\n   ⚠️  DÉPARTEMENTS À DOUBLE RISQUE ({len(double_risque)}) :")
for dept in double_risque.index:
    print(f"      • {dept}")

# =============================================================================
# 4. VISUALISATIONS
# =============================================================================
print("\n📈 [4/5] Génération des graphiques...")

# Vérification des données
if df.empty or 'Département' not in df.columns:
    raise ValueError("❌ Données insuffisantes pour générer les graphiques")

# --- GRAPHIQUE 1 : Top 20 départements ---
print("   → Graphique 1/3 : Top 20 départements...")
try:
    top20 = taux_dept.head(20).sort_values('Taux de passages aux urgences pour grippe', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = np.arange(len(top20))
    valeurs = top20['Taux de passages aux urgences pour grippe'].values
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top20)))

    bars = ax.barh(y_pos, valeurs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20.index, fontsize=9)
    ax.set_xlabel('Taux moyen passages urgences (pour 100k)', fontsize=11, fontweight='bold')
    ax.set_title('🚨 TOP 20 DÉPARTEMENTS - Pression urgences la + forte',
                fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, val in enumerate(valeurs):
        ax.text(val + 0.5, i, f'{val:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/07_top20_departements.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("      ✓ 07_top20_departements.png")
except Exception as e:
    print(f"      ❌ Erreur graphique 1 : {str(e)}")

# --- GRAPHIQUE 2 : Variabilité intra-régionale ---
print("   → Graphique 2/3 : Variabilité par région...")
try:
    top8_regions = df.groupby('Région')['Taux de passages aux urgences pour grippe'].mean().nlargest(8).index
    if len(top8_regions) > 0:
        df_top = df[df['Région'].isin(top8_regions)]

        fig, ax = plt.subplots(figsize=(16, 8))
        sns.boxplot(data=df_top, x='Région', y='Taux de passages aux urgences pour grippe',
                   palette='RdYlGn_r', ax=ax)

        ax.set_title('📊 VARIABILITÉ INTRA-RÉGIONALE - Top 8 régions',
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Région', fontsize=11, fontweight='bold')
        ax.set_ylabel('Taux passages urgences (pour 100k)', fontsize=11)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/08_variabilite_regionale.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("      ✓ 08_variabilite_regionale.png")
    else:
        print("      ⚠️  Pas assez de régions pour générer le graphique")
except Exception as e:
    print(f"      ❌ Erreur graphique 2 : {str(e)}")

# --- GRAPHIQUE 3 : Évolution départements critiques ---
print("   → Graphique 3/3 : Évolution départements critiques...")
try:
    if len(double_risque) > 0:
        fig, ax = plt.subplots(figsize=(16, 8))

        # Limiter à 5 départements pour éviter la surcharge
        for dept in double_risque.index[:5]:
            data = df[df['Département'] == dept].groupby('Date')['Taux de passages aux urgences pour grippe'].mean()
            if not data.empty:
                region = taux_dept.loc[dept, 'Région']
                ax.plot(data.index, data.values, label=f'{dept} ({region})',
                        linewidth=2.5, marker='o', markersize=2, alpha=0.85)

        if len(double_risque.index[:5]) > 0:  # Vérifier qu'au moins un département a été tracé
            ax.set_title('📈 ÉVOLUTION TEMPORELLE - Départements à double risque',
                        fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('Date', fontsize=11)
            ax.set_ylabel('Taux passages urgences (pour 100k)', fontsize=11)
            ax.legend(fontsize=10, loc='best', framealpha=0.95)
            ax.grid(alpha=0.3, linestyle='--')

            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/09_evolution_top_departements.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("      ✓ 09_evolution_top_departements.png")
        else:
            print("      ⚠️  Aucun département critique avec données suffisantes")
    else:
        print("      ⚠️  Aucun département à double risque identifié")
except Exception as e:
    print(f"      ❌ Erreur graphique 3 : {str(e)}")


# =============================================================================
# 5. INSIGHTS
# =============================================================================
print("\n💡 [5/5] Insights clés\n")
print("="*80)
print(" SYNTHÈSE DÉPARTEMENTALE ".center(80, "="))
print("="*80)

ratio = taux_dept['Taux de passages aux urgences pour grippe'].max() / taux_dept['Taux de passages aux urgences pour grippe'].min()
ecart = taux_dept['Taux de passages aux urgences pour grippe'].max() - taux_dept['Taux de passages aux urgences pour grippe'].min()

print(f"\n⚖️ DISPARITÉS DÉPARTEMENTALES :")
print(f"   • Ratio max/min : {ratio:.2f}x")
print(f"   • Écart max-min : {ecart:.1f} passages/100k")

print(f"\n🎯 CIBLAGE POUR ALLOCATION VACCINALE :")
print(f"   • {len(double_risque)} départements à double risque identifiés")
print(f"   • {len(top20)} départements dans le top 20 nécessitent une attention prioritaire")
print(f"   • {(variabilite > variabilite.quantile(0.75)).sum()} départements avec forte variabilité")

# Départements récents critiques
df_recent = df[df['Annee'] == df['Annee'].max()]
recent_critiques = df_recent.groupby('Département')['Taux de passages aux urgences pour grippe'].mean().nlargest(5)

print(f"\n🆕 DÉPARTEMENTS CRITIQUES EN {int(df['Annee'].max())} :")
for i, (dept, taux) in enumerate(recent_critiques.items(), 1):
    print(f"   {i}. {dept:30s} : {taux:.1f} pour 100k")

print("\n" + "="*80)
print("✅ ANALYSE DÉPARTEMENTALE TERMINÉE".center(80))
print("="*80)
print(f"🕐 Fin : {datetime.now().strftime('%H:%M:%S')}")
print(f"\n📊 3 graphiques générés dans {OUTPUT_DIR}/\n")