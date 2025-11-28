#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VACCIN IMPACT MAP - Analyse croisée
Fusionne données urgences + couverture vaccinale
Calcule les départements à fort impact
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" VACCIN IMPACT MAP - ANALYSE CROISÉE ".center(80, "="))
print("="*80)
print(f"🕐 Début : {datetime.now().strftime('%H:%M:%S')}\n")

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
print("📂 [1/5] Chargement des données...")

# Données urgences (votre analyse précédente)
df_urgences = pd.read_csv('data/grippe-passages-aux-urgences-et-actes-sos-medecins-departement.csv')
df_urgences['Date'] = pd.to_datetime(df_urgences['1er jour de la semaine'], errors='coerce')

# Données couverture vaccinale
df_vaccin = pd.read_csv('data/couvertures-vaccinales-des-adolescent-et-adultes-departement.csv', 
                        encoding='utf-8-sig')

print(f"   ✓ Urgences : {df_urgences.shape[0]} lignes")
print(f"   ✓ Vaccination : {df_vaccin.shape[0]} lignes")

# =============================================================================
# 2. PRÉPARATION DES DONNÉES
# =============================================================================
print("\n🧹 [2/5] Préparation et agrégation...")

# Taux moyen d'urgences par département (toute période)
urgences_dept = df_urgences.groupby('Département').agg({
    'Taux de passages aux urgences pour grippe': 'mean',
    'Taux d\'hospitalisations après passages aux urgences pour grippe': 'mean',
    'Région': 'first'
}).reset_index()

urgences_dept.columns = ['Département', 'Taux_Urgences_Moyen', 'Taux_Hospit', 'Région']

# Couverture vaccinale la plus récente par département
# Focus sur "Grippe 65 ans et plus" (population cible principale)
vaccin_recent = df_vaccin[df_vaccin['Année'] == df_vaccin['Année'].max()].copy()
vaccin_recent = vaccin_recent[['Département', 'Grippe 65 ans et plus', 'Année']].copy()
vaccin_recent.columns = ['Département', 'Couverture_65plus', 'Année']

print(f"   ✓ Année vaccination la + récente : {vaccin_recent['Année'].iloc[0]}")
print(f"   ✓ Départements avec données : {len(vaccin_recent)}")

# =============================================================================
# 3. FUSION DES DATASETS
# =============================================================================
print("\n🔗 [3/5] Fusion des données...")

df_fusion = urgences_dept.merge(vaccin_recent, on='Département', how='inner')

print(f"   ✓ Départements avec données complètes : {len(df_fusion)}")
print(f"   ✓ Couverture moyenne 65+ : {df_fusion['Couverture_65plus'].mean():.1f}%")

# Nettoyage des valeurs manquantes
df_fusion = df_fusion.dropna(subset=['Taux_Urgences_Moyen', 'Couverture_65plus'])

# =============================================================================
# 4. CALCUL DES SCORES D'IMPACT
# =============================================================================
print("\n📊 [4/5] Calcul des scores d'impact...")

# Score Impact corrigé (Option B - avec log, sans lag car données agrégées)
# Formule : Score_Impact = Taux_Urgences × log(1 + (100 - Couverture))
# Le log modère le poids lorsque la couverture est faible (relation non-linéaire)
df_fusion['Score_Impact'] = (
    df_fusion['Taux_Urgences_Moyen'] *
    np.log(1 + (100 - df_fusion['Couverture_65plus']))
).round(1)

# === GAP_VACCINAL CORRIGÉ (avec composante régionale) ===
# Calcul de la moyenne nationale
couverture_moyenne = df_fusion['Couverture_65plus'].mean()
df_fusion['Gap_National'] = (couverture_moyenne - df_fusion['Couverture_65plus']).round(1)

# Calcul de la moyenne régionale (par région)
moyennes_regionales = df_fusion.groupby('Région')['Couverture_65plus'].transform('mean')
df_fusion['Gap_Regional'] = (moyennes_regionales - df_fusion['Couverture_65plus']).round(1)

# Gap vaccinal corrigé : moyenne des deux composantes
df_fusion['Gap_Vaccinal'] = ((df_fusion['Gap_National'] + df_fusion['Gap_Regional']) / 2).round(1)

# === CLASSIFICATION PAR TYPE DE ZONE ===
def classifier_type_zone(row):
    """Classifie le département par type de zone"""
    dept = row['Département']
    taux_urg = row['Taux_Urgences_Moyen']

    # Départements urbains denses (grandes métropoles identifiées par nom)
    urbains_denses_noms = ['Paris', 'Hauts-de-Seine', 'Seine-Saint-Denis', 'Val-de-Marne',
                           'Rhône', 'Bouches-du-Rhône', 'Nord', 'Gironde', 'Haute-Garonne', 'Loire-Atlantique']

    if any(nom in dept for nom in urbains_denses_noms):
        return 'Urbain dense'
    elif taux_urg > 100:
        return 'Urbain'
    elif taux_urg > 50:
        return 'Mixte'
    else:
        return 'Rural'

df_fusion['Type_Zone'] = df_fusion.apply(classifier_type_zone, axis=1)

# === POTENTIEL_RÉDUCTION_URGENCES CORRIGÉ (coefficients zonaux) ===
coef_par_zone = {
    'Urbain dense': -0.85,
    'Urbain': -0.70,
    'Mixte': -0.60,
    'Rural': -0.45
}

df_fusion['Coef_Regional'] = df_fusion['Type_Zone'].map(coef_par_zone)

df_fusion['Potentiel_Reduction_Urgences'] = (
    df_fusion['Gap_Vaccinal'] * df_fusion['Coef_Regional']
).abs().round(1)

print(f"   ✓ Gap vaccinal (National + Régional)/2 : {df_fusion['Gap_Vaccinal'].mean():.1f} pts")
print(f"   ✓ Classification zonale : {df_fusion['Type_Zone'].value_counts().to_dict()}")
print(f"   ✓ Potentiel réduction moyen : {df_fusion['Potentiel_Reduction_Urgences'].mean():.1f} urgences/100k")

# === CATÉGORISATION DU RISQUE CORRIGÉE (quantiles dynamiques) ===
# Utilisation des quartiles de la distribution réelle
try:
    df_fusion['Catégorie_Risque'] = pd.qcut(
        df_fusion['Score_Impact'],
        q=4,
        labels=['🟢 Faible', '🟡 Moyen', '🟠 Élevé', '🔴 Critique'],
        duplicates='drop'  # Gérer les valeurs identiques
    )
    print("   ✓ Catégorie_Risque : quartiles dynamiques (Q1, Q2, Q3)")
except ValueError:  # Si pas assez de valeurs uniques
    df_fusion['Catégorie_Risque'] = pd.cut(
        df_fusion['Score_Impact'],
        bins=[0, 250, 500, 750, float('inf')],
        labels=['🟢 Faible', '🟡 Moyen', '🟠 Élevé', '🔴 Critique']
    )
    print("   ⚠️  Catégorie_Risque : seuils fixes (pas assez de valeurs uniques pour quartiles)")

# Tri par score décroissant
df_fusion = df_fusion.sort_values('Score_Impact', ascending=False).reset_index(drop=True)

print(f"   ✓ Score impact moyen : {df_fusion['Score_Impact'].mean():.1f}")
print(f"   ✓ Distribution Score_Impact : [{df_fusion['Score_Impact'].min():.0f}, {df_fusion['Score_Impact'].quantile(0.5):.0f}, {df_fusion['Score_Impact'].max():.0f}]")

# =============================================================================
# 5. RÉSULTATS & INSIGHTS
# =============================================================================
print("\n💡 [5/5] Résultats\n")
print("="*80)
print(" TOP 15 DÉPARTEMENTS À PRIORISER ".center(80, "="))
print("="*80)

print(f"\n{'#':<4} {'Département':<25} {'Score':<8} {'Urg/100k':<10} {'Vacc %':<8} {'Gap':<6} Risque")
print("-"*80)

for i, row in df_fusion.head(15).iterrows():
    print(f"{i+1:<4} {row['Département']:<25} {row['Score_Impact']:<8.1f} "
          f"{row['Taux_Urgences_Moyen']:<10.1f} {row['Couverture_65plus']:<8.1f} "
          f"{row['Gap_Vaccinal']:<6.1f} {row['Catégorie_Risque']}")

# Statistiques par catégorie
print("\n" + "="*80)
print(" RÉPARTITION PAR NIVEAU DE RISQUE ".center(80, "="))
print("="*80 + "\n")

for categorie in ['🔴 Critique', '🟠 Élevé', '🟡 Moyen', '🟢 Faible']:
    count = (df_fusion['Catégorie_Risque'] == categorie).sum()
    pct = (count / len(df_fusion) * 100)
    print(f"   {categorie:<15} : {count:3d} départements ({pct:5.1f}%)")

# Corrélation
corr = df_fusion['Taux_Urgences_Moyen'].corr(df_fusion['Couverture_65plus'])
print(f"\n🔗 Corrélation Urgences ↔ Vaccination : r = {corr:.3f}")
if corr < -0.3:
    print("   → Corrélation négative : + de vaccination = - d'urgences ✓")
else:
    print("   → Pas de corrélation claire (autres facteurs en jeu)")

# =============================================================================
# 6. VISUALISATION
# =============================================================================
print("\n📈 [6/6] Génération des graphiques...")

# GRAPHIQUE 1 : Scatter plot Impact
fig, ax = plt.subplots(figsize=(14, 8))

colors = {'🔴 Critique': 'darkred', '🟠 Élevé': 'orange', 
          '🟡 Moyen': 'gold', '🟢 Faible': 'green'}

for categorie, color in colors.items():
    subset = df_fusion[df_fusion['Catégorie_Risque'] == categorie]
    ax.scatter(subset['Couverture_65plus'], subset['Taux_Urgences_Moyen'],
               s=subset['Score_Impact']*0.5, alpha=0.6, c=color, 
               label=categorie, edgecolors='black', linewidth=0.5)

# Annotations pour top 10
for i, row in df_fusion.head(10).iterrows():
    ax.annotate(row['Département'], 
                (row['Couverture_65plus'], row['Taux_Urgences_Moyen']),
                fontsize=8, alpha=0.8)

ax.set_xlabel('Couverture vaccinale 65+ (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Taux moyen passages urgences (pour 100k)', fontsize=12, fontweight='bold')
ax.set_title('🎯 VACCIN IMPACT MAP - Départements à prioriser\n(Taille des bulles = Score d\'impact)', 
             fontsize=14, fontweight='bold', pad=15)
ax.legend(title='Niveau de risque', fontsize=10, loc='upper right')
ax.grid(alpha=0.3, linestyle='--')

# Ligne de tendance
z = np.polyfit(df_fusion['Couverture_65plus'], df_fusion['Taux_Urgences_Moyen'], 1)
p = np.poly1d(z)
ax.plot(df_fusion['Couverture_65plus'], p(df_fusion['Couverture_65plus']), 
        "r--", alpha=0.5, linewidth=2, label='Tendance')

plt.tight_layout()
plt.savefig('output/10_vaccin_impact_map.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 10_vaccin_impact_map.png")

# GRAPHIQUE 2 : Top 20 scores
fig, ax = plt.subplots(figsize=(12, 10))

top20 = df_fusion.head(20)
colors_bar = [colors[cat] for cat in top20['Catégorie_Risque']]

y_pos = np.arange(len(top20))
bars = ax.barh(y_pos, top20['Score_Impact'], color=colors_bar, 
               edgecolor='black', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(top20['Département'], fontsize=9)
ax.set_xlabel('Score d\'Impact', fontsize=11, fontweight='bold')
ax.set_title('🎯 TOP 20 DÉPARTEMENTS - Score d\'impact vaccination', 
             fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Valeurs sur barres
for i, val in enumerate(top20['Score_Impact']):
    ax.text(val + 10, i, f'{val:.0f}', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('output/11_top20_impact_scores.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 11_top20_impact_scores.png")


# =============================================================================
# GRAPHIQUE 3 : CARTE DE FRANCE PAR DÉPARTEMENT
# =============================================================================
print("\n🗺️ Création de la carte de France...")

import plotly.graph_objects as go
import json
import urllib.request

# Fonction pour normaliser les codes départements
def normaliser_code_dept(code):
    """Convertit les codes en format à 2 chiffres (01, 02, etc.)"""
    if pd.isna(code):
        return None
    code_str = str(code).strip()
    
    # Si c'est déjà un nombre, le formater
    if code_str.isdigit():
        return code_str.zfill(2)  # Ajoute un 0 devant si nécessaire
    
    # Si format "(XX)", extraire le code
    if '(' in code_str and ')' in code_str:
        extracted = code_str.split('(')[1].split(')')[0].strip()
        if extracted.isdigit():
            return extracted.zfill(2)
    
    # Départements spéciaux (Corse, DOM-TOM)
    special_codes = {
        '2A': '2A', '2B': '2B',  # Corse
        '971': '971', '972': '972', '973': '973', '974': '974', '976': '976'  # DOM
    }
    if code_str in special_codes:
        return special_codes[code_str]
    
    return None

# CHARGEMENT DES DONNÉES
print("📊 Chargement des données...")

# Supposons que df_fusion existe déjà, sinon le recréer :
# df_fusion = pd.read_csv('votre_fichier_fusion.csv')

# EXEMPLE avec vos données de grippe et vaccination
df_dept = pd.read_csv('data/grippe-passages-aux-urgences-et-actes-sos-medecins-departement.csv')
df_vacc = pd.read_csv('data/couvertures-vaccinales-des-adolescent-et-adultes-departement.csv')

# Préparer les données pour la carte
print("🔧 Normalisation des codes départements...")

# Nettoyer et normaliser les codes
df_dept['Code_Dept_Clean'] = df_dept['Département Code'].apply(normaliser_code_dept)
df_vacc['Code_Dept_Clean'] = df_vacc['Département Code'].apply(normaliser_code_dept)

# Calculer un score d'impact simplifié (moyenne par département)
# Note: Adapter selon votre logique de calcul
df_urgences_agg = df_dept.groupby('Code_Dept_Clean').agg({
    'Taux de passages aux urgences pour grippe': 'mean',
    'Département': 'first'
}).reset_index()

df_vacc_recent = df_vacc[df_vacc['Année'] == df_vacc['Année'].max()]
df_vacc_agg = df_vacc_recent.groupby('Code_Dept_Clean').agg({
    'Grippe 65 ans et plus': 'mean'
}).reset_index()

# Fusionner
df_map = df_urgences_agg.merge(df_vacc_agg, on='Code_Dept_Clean', how='left')

# Calculer le score d'impact (formule simplifiée)
# Score élevé = Urgences élevées + Couverture faible
df_map['Couverture_65plus'] = df_map['Grippe 65 ans et plus'].fillna(df_map['Grippe 65 ans et plus'].median())
df_map['Urgences'] = df_map['Taux de passages aux urgences pour grippe'].fillna(0)

# Normaliser entre 0 et 100
urgences_norm = (df_map['Urgences'] - df_map['Urgences'].min()) / (df_map['Urgences'].max() - df_map['Urgences'].min()) * 100
couv_norm = 100 - df_map['Couverture_65plus']  # Inverser : faible couverture = score élevé

df_map['Score_Impact'] = (urgences_norm * 0.6 + couv_norm * 0.4)

print(f"✓ {len(df_map)} départements préparés")
print(f"  Codes disponibles: {df_map['Code_Dept_Clean'].nunique()}")
print(f"  Exemple codes: {df_map['Code_Dept_Clean'].head(10).tolist()}")

# CHARGEMENT DU GEOJSON
print("\n🗺️ Chargement du GeoJSON...")
geojson_url = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"

try:
    with urllib.request.urlopen(geojson_url) as url:
        departements_geojson = json.loads(url.read().decode())
    
    # Vérifier les codes dans le GeoJSON
    geojson_codes = [feat['properties']['code'] for feat in departements_geojson['features']]
    print(f"✓ GeoJSON chargé avec {len(geojson_codes)} départements")
    print(f"  Exemple codes GeoJSON: {geojson_codes[:10]}")
    
    # DIAGNOSTIC : Codes manquants
    codes_data = set(df_map['Code_Dept_Clean'].dropna())
    codes_geojson = set(geojson_codes)
    
    manquants_data = codes_geojson - codes_data
    manquants_geojson = codes_data - codes_geojson
    
    if manquants_geojson:
        print(f"⚠️ Codes dans vos données absents du GeoJSON: {manquants_geojson}")
    if manquants_data:
        print(f"⚠️ Codes GeoJSON sans données: {manquants_data}")
    
    print(f"✓ Match: {len(codes_data & codes_geojson)} départements avec données ET géométrie")
    
    # CRÉATION DE LA CARTE
    print("\n🎨 Création de la carte interactive...")
    
    fig = go.Figure(go.Choroplethmapbox(
        geojson=departements_geojson,
        locations=df_map['Code_Dept_Clean'],
        z=df_map['Score_Impact'],
        featureidkey="properties.code",  # CLÉ CRITIQUE : doit correspondre aux codes normalisés
        colorscale=[
            [0, '#2ecc71'],      # Vert : faible priorité
            [0.33, '#f1c40f'],   # Jaune
            [0.66, '#e67e22'],   # Orange
            [1, '#c0392b']       # Rouge foncé : haute priorité
        ],
        marker_opacity=0.75,
        marker_line_width=1,
        marker_line_color='white',
        colorbar=dict(
            title="Score<br>Impact<br>(0-100)",
            thickness=20,
            len=0.7,
            x=1.02
        ),
        text=df_map['Département'],
        customdata=df_map[['Urgences', 'Couverture_65plus']],
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'Score Impact: %{z:.1f}/100<br>' +
            'Taux urgences: %{customdata[0]:.2f}<br>' +
            'Couverture 65+: %{customdata[1]:.1f}%<br>' +
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=4.8,
        mapbox_center={"lat": 46.8, "lon": 2.5},
        title={
            'text': '🗺️ SCORE D\'IMPACT VACCINATION GRIPPE PAR DÉPARTEMENT<br>' +
                    '<sub>Plus le score est élevé, plus la priorité d\'action est forte</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial Black', 'color': '#2c3e50'}
        },
        height=900,
        width=1400,
        margin={"r":50,"t":100,"l":0,"b":0},
        font=dict(family="Arial", size=12)
    )
    
    # Sauvegarder
    fig.write_html('output/carte_france_vaccination_corrigee.html')
    print("✅ CARTE CRÉÉE : carte_france_vaccination_corrigee.html")
    
    # Afficher les top départements prioritaires
    print("\n📊 TOP 10 DÉPARTEMENTS PRIORITAIRES :")
    top_10 = df_map.nlargest(10, 'Score_Impact')[['Département', 'Score_Impact', 'Urgences', 'Couverture_65plus']]
    print(top_10.to_string(index=False))
    
except Exception as e:
    print(f"❌ ERREUR : {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("POINTS DE VIGILANCE :")
print("="*80)

# Carte en barres horizontales par région
colors_region = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(df_region)))
y_pos = np.arange(len(df_region))

bars = ax.barh(y_pos, df_region['Score_Impact'], color=colors_region, 
               edgecolor='black', linewidth=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(df_region['Région'], fontsize=10)
ax.set_xlabel('Score d\'Impact Moyen', fontsize=12, fontweight='bold')
ax.set_title('🗺️ SCORE D\'IMPACT PAR RÉGION\n(Moyenne des départements de la région)', 
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Valeurs sur barres
for i, val in enumerate(df_region['Score_Impact']):
    ax.text(val + 5, i, f'{val:.0f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('output/12_carte_regions_impact.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 12_carte_regions_impact.png")


# =============================================================================
# GRAPHIQUE 4 : MATRICE GAP VACCINAL vs URGENCES
# =============================================================================
print("\n📊 Création de la matrice gap vaccinal...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('📊 ANALYSE MULTI-DIMENSIONNELLE - Impact Vaccination', 
             fontsize=16, fontweight='bold', y=0.995)

# --- SUBPLOT 1 : Distribution des scores ---
ax1 = axes[0, 0]
colors_hist = {'🔴 Critique': 'darkred', '🟠 Élevé': 'orange', 
               '🟡 Moyen': 'gold', '🟢 Faible': 'green'}

for categorie, color in colors_hist.items():
    subset = df_fusion[df_fusion['Catégorie_Risque'] == categorie]
    ax1.hist(subset['Score_Impact'], bins=15, alpha=0.6, color=color, 
             label=categorie, edgecolor='black', linewidth=0.5)

ax1.set_xlabel('Score d\'Impact', fontsize=11, fontweight='bold')
ax1.set_ylabel('Nombre de départements', fontsize=11, fontweight='bold')
ax1.set_title('Distribution des scores d\'impact', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3, linestyle='--')

# --- SUBPLOT 2 : Gap vaccinal par région ---
ax2 = axes[0, 1]

gap_region = df_fusion.groupby('Région')['Gap_Vaccinal'].mean().sort_values()
colors_gap = ['green' if x < 0 else 'red' for x in gap_region.values]

ax2.barh(range(len(gap_region)), gap_region.values, color=colors_gap, 
         edgecolor='black', linewidth=0.5, alpha=0.7)
ax2.set_yticks(range(len(gap_region)))
ax2.set_yticklabels(gap_region.index, fontsize=9)
ax2.set_xlabel('Gap vaccinal vs moyenne nationale (points %)', fontsize=10, fontweight='bold')
ax2.set_title('Gap vaccinal par région', fontsize=12, fontweight='bold')
ax2.axvline(0, color='black', linestyle='--', linewidth=1.5)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# --- SUBPLOT 3 : Corrélation Score vs Hospitalisation ---
ax3 = axes[1, 0]

scatter = ax3.scatter(df_fusion['Score_Impact'], 
                      df_fusion['Taux_Hospit'],
                      s=100, alpha=0.6, 
                      c=df_fusion['Couverture_65plus'],
                      cmap='RdYlGn', edgecolors='black', linewidth=0.5)

# Annotations top 5
for i, row in df_fusion.head(5).iterrows():
    ax3.annotate(row['Département'], 
                 (row['Score_Impact'], row['Taux_Hospit']),
                 fontsize=8, alpha=0.8)

ax3.set_xlabel('Score d\'Impact', fontsize=11, fontweight='bold')
ax3.set_ylabel('Taux d\'hospitalisation (%)', fontsize=11, fontweight='bold')
ax3.set_title('Impact vs Hospitalisation (couleur = couverture)', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3, linestyle='--')

cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Couverture 65+ (%)', fontsize=9)

# --- SUBPLOT 4 : Top 10 départements (tableau) ---
ax4 = axes[1, 1]
ax4.axis('off')

top10 = df_fusion.head(10)[['Département', 'Score_Impact', 'Taux_Urgences_Moyen', 
                             'Couverture_65plus', 'Gap_Vaccinal']]

# Créer un tableau
table_data = []
table_data.append(['#', 'Département', 'Score', 'Urg/100k', 'Vacc%', 'Gap'])

for i, row in top10.iterrows():
    table_data.append([
        str(i+1),
        row['Département'][:20],
        f"{row['Score_Impact']:.0f}",
        f"{row['Taux_Urgences_Moyen']:.0f}",
        f"{row['Couverture_65plus']:.1f}",
        f"{row['Gap_Vaccinal']:.1f}"
    ])

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.08, 0.35, 0.12, 0.15, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style du header
for i in range(6):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Couleurs alternées
for i in range(1, len(table_data)):
    color = '#f0f0f0' if i % 2 == 0 else 'white'
    for j in range(6):
        table[(i, j)].set_facecolor(color)

ax4.set_title('🏆 TOP 10 DÉPARTEMENTS PRIORITAIRES', 
              fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('output/13_analyse_multidimensionnelle.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 13_analyse_multidimensionnelle.png")

# =============================================================================
# GRAPHIQUE 5 : SIMULATION AVANT/APRÈS CIBLAGE
# =============================================================================
print("\n💰 Création du graphique ROI avant/après...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('💰 SIMULATION : Impact du Ciblage Géographique', 
             fontsize=16, fontweight='bold')

# Scénario 1 : Sans ciblage (distribution uniforme)
doses_total = 100000
impact_sans_ciblage = df_fusion['Taux_Urgences_Moyen'].mean() * 0.0012  # 0.12% réduction par dose
urgences_evitees_sans = doses_total * impact_sans_ciblage

# Scénario 2 : Avec ciblage (focus top 15)
top15 = df_fusion.head(15)
impact_avec_ciblage = top15['Taux_Urgences_Moyen'].mean() * 0.0018  # 0.18% réduction par dose
urgences_evitees_avec = doses_total * impact_avec_ciblage

# --- SUBPLOT 1 : Barres comparatives ---
scenarios = ['Sans ciblage\n(distribution uniforme)', 'Avec ciblage\n(top 15 départements)']
urgences = [urgences_evitees_sans, urgences_evitees_avec]
colors_bar = ['#ff6b6b', '#51cf66']

bars = ax1.bar(scenarios, urgences, color=colors_bar, edgecolor='black', 
               linewidth=2, alpha=0.8, width=0.6)

# Valeurs sur barres
for i, (bar, val) in enumerate(zip(bars, urgences)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.0f}\nurgences\névitées',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Flèche de gain
ax1.annotate('', xy=(1, urgences_evitees_avec), xytext=(0, urgences_evitees_sans),
             arrowprops=dict(arrowstyle='->', lw=3, color='green'))
ax1.text(0.5, (urgences_evitees_sans + urgences_evitees_avec)/2,
         f'+{((urgences_evitees_avec/urgences_evitees_sans - 1)*100):.0f}%',
         ha='center', fontsize=16, fontweight='bold', color='green',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2))

ax1.set_ylabel('Urgences évitées', fontsize=12, fontweight='bold')
ax1.set_title('Efficacité des campagnes (100k doses)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(urgences) * 1.3)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# --- SUBPLOT 2 : ROI financier ---
cout_passage = 200  # euros
cout_dose = 15  # euros

cout_campagne = doses_total * cout_dose
economie_sans = urgences_evitees_sans * cout_passage
economie_avec = urgences_evitees_avec * cout_passage

roi_sans = (economie_sans - cout_campagne) / cout_campagne * 100
roi_avec = (economie_avec - cout_campagne) / cout_campagne * 100

categories = ['Coût\ncampagne', 'Économies\nsans ciblage', 'Économies\navec ciblage']
montants = [cout_campagne/1000, economie_sans/1000, economie_avec/1000]
colors_roi = ['red', 'orange', 'green']

bars2 = ax2.bar(categories, montants, color=colors_roi, edgecolor='black',
                linewidth=2, alpha=0.8, width=0.6)

# Valeurs
for bar, val in zip(bars2, montants):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.0f}k€',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# ROI
ax2.text(1.5, max(montants) * 0.9,
         f'ROI avec ciblage:\n+{roi_avec:.0f}%',
         ha='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', 
                   edgecolor='darkgreen', linewidth=2))

ax2.set_ylabel('Montant (milliers €)', fontsize=12, fontweight='bold')
ax2.set_title('Retour sur investissement', fontsize=13, fontweight='bold')
ax2.set_ylim(0, max(montants) * 1.2)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('output/14_simulation_roi.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ 14_simulation_roi.png")

print("\n" + "="*80)
print(f"✅ 5 VISUALISATIONS GÉNÉRÉES".center(80))
print("="*80)



# =============================================================================
# 7. EXPORT CSV POUR UTILISATION ULTÉRIEURE
# =============================================================================
print("\n📁 Export des résultats...")

df_fusion.to_csv('output/departements_scores_impact.csv', index=False, encoding='utf-8-sig')
print("   ✓ departements_scores_impact.csv")

# =============================================================================
# 8. RECOMMANDATIONS STRATÉGIQUES
# =============================================================================
print("\n" + "="*80)
print(" RECOMMANDATIONS STRATÉGIQUES ".center(80, "="))
print("="*80)

# Départements critiques
critiques = df_fusion[df_fusion['Catégorie_Risque'] == '🔴 Critique']
print(f"\n🔴 {len(critiques)} DÉPARTEMENTS À RISQUE CRITIQUE")
print(f"   → Représentent {(critiques['Taux_Urgences_Moyen'].sum() / df_fusion['Taux_Urgences_Moyen'].sum() * 100):.1f}% de la charge nationale")
print(f"   → Couverture moyenne : {critiques['Couverture_65plus'].mean():.1f}% (vs {couverture_moyenne:.1f}% national)")

# Potentiel d'amélioration
doses_necessaires = (critiques['Gap_Vaccinal'] * 1000).sum()  # Estimation simplifiée
print(f"\n💉 POTENTIEL D'AMÉLIORATION")
print(f"   → Combler le gap des départements critiques : ~{doses_necessaires:,.0f} doses")
print(f"   → Réduction passages urgences estimée : {(doses_necessaires * 0.05):.0f} par semaine")
print(f"   → ROI estimé : {(doses_necessaires * 0.05 * 200):.0f}€ économisés/semaine")

print("\n" + "="*80)
print("✅ ANALYSE TERMINÉE".center(80))
print("="*80)
print(f"🕐 Fin : {datetime.now().strftime('%H:%M:%S')}")
print(f"\n📊 2 nouveaux graphiques générés dans ../output/\n")