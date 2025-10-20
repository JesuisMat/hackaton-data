#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALYSE CROISÉE MULTI-NIVEAUX
Objectif : Corrélations, cohérence, insights synthétiques
Durée estimée : 5-8 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print(" ANALYSE CROISÉE - SYNTHÈSE MULTI-NIVEAUX ".center(80, "="))
print("="*80)
print(f"\n🕐 Début : {datetime.now().strftime('%H:%M:%S')}\n")

# =============================================================================
# 1. CHARGEMENT DES 3 NIVEAUX
# =============================================================================
print("📂 ÉTAPE 1/4 : Chargement des données multi-niveaux")
print("-"*80)

df_nat = pd.read_csv('/mnt/user-data/uploads/grippepassagesauxurgencesetactessosmedecinsfrance.csv')
df_reg = pd.read_csv('/mnt/user-data/uploads/grippepassagesurgencesetactessosmedecin_reg.csv')
df_dept = pd.read_csv('/mnt/user-data/uploads/grippepassagesauxurgencesetactessosmedecinsdepartement.csv')

for df in [df_nat, df_reg, df_dept]:
    df['Date'] = pd.to_datetime(df['1er jour de la semaine'], errors='coerce')

print(f"✅ National : {df_nat.shape[0]} observations")
print(f"✅ Régional : {df_reg.shape[0]} observations ({df_reg['Région'].nunique()} régions)")
print(f"✅ Départemental : {df_dept.shape[0]} observations ({df_dept['Département'].nunique()} départements)")

# =============================================================================
# 2. CORRÉLATIONS URGENCES ↔ SOS MÉDECINS
# =============================================================================
print("\n" + "="*80)
print("🔗 ÉTAPE 2/4 : Analyse des corrélations")
print("-"*80)

# Corrélation nationale
corr_nat = df_nat['Taux de passages aux urgences pour grippe'].corr(
    df_nat['Taux d\'actes médicaux SOS médecins pour grippe']
)

print(f"\n🇫🇷 CORRÉLATION NATIONALE Urgences ↔ SOS Médecins : r = {corr_nat:.3f}")

# Corrélations régionales
corr_reg = df_reg.groupby('Région').apply(
    lambda x: x['Taux de passages aux urgences pour grippe'].corr(
        x['Taux d\'actes médicaux SOS médecins pour grippe']
    )
).sort_values(ascending=False)

print(f"\n📍 TOP 5 RÉGIONS (forte corrélation) :")
for i, (region, corr) in enumerate(corr_reg.head(5).items(), 1):
    print(f"   {i}. {region:35s} : r = {corr:.3f}")

print(f"\n⚠️  RÉGIONS AVEC FAIBLE CORRÉLATION (<0.3) :")
faibles = corr_reg[corr_reg < 0.3]
if len(faibles) > 0:
    for region, corr in faibles.items():
        print(f"   • {region:35s} : r = {corr:.3f} → Possible problème d'accès aux soins")
else:
    print("   → Aucune région identifiée")

# =============================================================================
# 3. VISUALISATIONS
# =============================================================================
print("\n" + "="*80)
print("📊 ÉTAPE 3/4 : Génération des visualisations")
print("-"*80)

# GRAPHIQUE 1 : Scatter plots corrélation
print("\n⏳ Création du graphique 1/3 : Corrélations Urgences/SOS...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('🔗 CORRÉLATION URGENCES ↔ SOS MÉDECINS', fontsize=14, fontweight='bold')

# National
axes[0].scatter(df_nat['Taux de passages aux urgences pour grippe'],
                df_nat['Taux d\'actes médicaux SOS médecins pour grippe'],
                alpha=0.5, s=40, c=df_nat['Date'].dt.month, cmap='coolwarm', edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('Taux passages urgences (pour 100k)', fontsize=10)
axes[0].set_ylabel('Taux actes SOS Médecins (pour 100k)', fontsize=10)
axes[0].set_title(f'🇫🇷 National (r={corr_nat:.3f})', fontsize=11, fontweight='bold')
axes[0].grid(alpha=0.3, linestyle='--')

# Régional
axes[1].scatter(df_reg['Taux de passages aux urgences pour grippe'],
                df_reg['Taux d\'actes médicaux SOS médecins pour grippe'],
                alpha=0.3, s=15, color='steelblue', edgecolors='black', linewidth=0.3)
axes[1].set_xlabel('Taux passages urgences (pour 100k)', fontsize=10)
axes[1].set_ylabel('Taux actes SOS Médecins (pour 100k)', fontsize=10)
axes[1].set_title('📍 Régional (toutes régions)', fontsize=11, fontweight='bold')
axes[1].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/10_correlations_urgences_sos.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé : 10_correlations_urgences_sos.png")

# GRAPHIQUE 2 : Évolution comparée niveaux
print("⏳ Création du graphique 2/3 : Évolution multi-niveaux...")

# Moyennes par année
tendance_nat = df_nat.groupby(df_nat['Date'].dt.year)['Taux de passages aux urgences pour grippe'].mean()

top_region = df_reg.groupby('Région')['Taux de passages aux urgences pour grippe'].mean().idxmax()
bottom_region = df_reg.groupby('Région')['Taux de passages aux urgences pour grippe'].mean().idxmin()

tendance_top_reg = df_reg[df_reg['Région'] == top_region].groupby(
    df_reg[df_reg['Région'] == top_region]['Date'].dt.year
)['Taux de passages aux urgences pour grippe'].mean()

tendance_bottom_reg = df_reg[df_reg['Région'] == bottom_region].groupby(
    df_reg[df_reg['Région'] == bottom_region]['Date'].dt.year
)['Taux de passages aux urgences pour grippe'].mean()

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(tendance_nat.index, tendance_nat.values, marker='o', markersize=8,
        linewidth=3, label='🇫🇷 National', color='black')
ax.plot(tendance_top_reg.index, tendance_top_reg.values, marker='s', markersize=7,
        linewidth=2.5, label=f'🔴 {top_region}', color='red')
ax.plot(tendance_bottom_reg.index, tendance_bottom_reg.values, marker='^', markersize=7,
        linewidth=2.5, label=f'🟢 {bottom_region}', color='green')

ax.set_title('📈 ÉVOLUTION ANNUELLE COMPARÉE - National vs Régions extrêmes', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Année', fontsize=11, fontweight='bold')
ax.set_ylabel('Taux moyen passages urgences (pour 100k)', fontsize=11)
ax.legend(fontsize=10, loc='best', framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/11_evolution_comparee.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé : 11_evolution_comparee.png")

# GRAPHIQUE 3 : Matrice de corrélation
print("⏳ Création du graphique 3/3 : Matrice de corrélation...")

# Matrice sur données nationales
colonnes = [
    'Taux de passages aux urgences pour grippe',
    'Taux d\'hospitalisations après passages aux urgences pour grippe',
    'Taux d\'actes médicaux SOS médecins pour grippe'
]

corr_matrix = df_nat[colonnes].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=2, cbar_kws={'label': 'Coefficient de corrélation'},
            vmin=-1, vmax=1, ax=ax)

ax.set_title('🔗 MATRICE DE CORRÉLATION - Indicateurs nationaux', 
             fontsize=13, fontweight='bold', pad=15)

# Noms plus courts pour lisibilité
labels = ['Urgences', 'Hospitalisations', 'SOS Médecins']
ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
ax.set_yticklabels(labels, fontsize=10, rotation=0)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/12_matrice_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Sauvegardé : 12_matrice_correlation.png")

# =============================================================================
# 4. SYNTHÈSE INSIGHTS
# =============================================================================
print("\n" + "="*80)
print("💡 ÉTAPE 4/4 : Synthèse des insights multi-niveaux")
print("-"*80)

# Calculs pour synthèse
taux_reg_moy = df_reg.groupby('Région')['Taux de passages aux urgences pour grippe'].mean()
ratio_reg = taux_reg_moy.max() / taux_reg_moy.min()

taux_dept_moy = df_dept.groupby('Département')['Taux de passages aux urgences pour grippe'].mean()
ratio_dept = taux_dept_moy.max() / taux_dept_moy.min()

hospit_nat = df_nat.groupby('Classe d\'âge')['Taux d\'hospitalisations après passages aux urgences pour grippe'].mean()

print(f"""
🗺️  DISPARITÉS TERRITORIALES :
   • Ratio région max/min : {ratio_reg:.2f}x
   • Ratio département max/min : {ratio_dept:.2f}x
   → Forte hétérogénéité nécessitant ciblage fin

🏥 HOSPITALISATION :
   • Classe la + hospitalisée : {hospit_nat.idxmax()} ({hospit_nat.max():.2f}%)
   • Classe la - hospitalisée : {hospit_nat.idxmin()} ({hospit_nat.min():.2f}%)
   → Prioriser vaccination des populations vulnérables

🚑 ACCÈS AUX SOINS :
   • Corrélation nationale Urgences/SOS : {corr_nat:.3f}
   • {len(faibles)} régions avec faible corrélation
   → Possibles problèmes d'accès aux soins primaires

🎯 RECOMMANDATIONS :
   1. Cibler les top 20 départements à risque
   2. Renforcer la vaccination 65+ ans (+ hospitalisés)
   3. Anticiper les pics janvier-février (commandes nov-déc)
   4. Améliorer accès SOS Médecins dans régions à faible corrélation
""")

print("="*80)
print("✅ ANALYSE CROISÉE TERMINÉE".center(80))
print("="*80)
print(f"🕐 Fin : {datetime.now().strftime('%H:%M:%S')}")
print(f"\n📊 3 visualisations générées dans /mnt/user-data/outputs/\n")