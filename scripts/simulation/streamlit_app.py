#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🚀 DASHBOARD COMPLET - 5 PAGES FONCTIONNELLES
Hackathon Stratégie Vaccinale Grippe - Version Finale
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="🦠 Stratégie Vaccinale Grippe",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    .stMetric {
        background: grey;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .badge-critique { background: #dc3545; color: white; padding: 0.25rem 0.75rem; 
                      border-radius: 15px; font-weight: bold; font-size: 0.85rem; display: inline-block; }
    .badge-eleve { background: #fd7e14; color: white; padding: 0.25rem 0.75rem; 
                   border-radius: 15px; font-weight: bold; font-size: 0.85rem; display: inline-block; }
    .badge-moyen { background: #ffc107; color: #333; padding: 0.25rem 0.75rem; 
                   border-radius: 15px; font-weight: bold; font-size: 0.85rem; display: inline-block; }
    .badge-faible { background: #28a745; color: white; padding: 0.25rem 0.75rem; 
                    border-radius: 15px; font-weight: bold; font-size: 0.85rem; display: inline-block; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
    section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# Initialisation des modèles dans session_state
if 'model_temp' not in st.session_state:
    st.session_state.model_temp = None
if 'model_doses' not in st.session_state:
    st.session_state.model_doses = None
# =============================================================================
# FONCTIONS DE CHARGEMENT
# =============================================================================

@st.cache_data(ttl=3600)
def load_all_data():
    """Charge tous les datasets"""
    try:
        data_dir = Path("data")
        
        df_france = pd.read_csv(data_dir / "grippe-passages-aux-urgences-et-actes-sos-medecins-france.csv")
        df_regions = pd.read_csv(data_dir / "grippe-passages-urgences-et-actes-sos-medecin_reg.csv")
        df_departements = pd.read_csv(data_dir / "grippe-passages-aux-urgences-et-actes-sos-medecins-departement.csv")
        
        df_vacc_france = pd.read_csv(data_dir / "couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-france.csv")
        df_vacc_regions = pd.read_csv(data_dir / "couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-region.csv")
        df_vacc_depts = pd.read_csv(data_dir / "couvertures-vaccinales-des-adolescent-et-adultes-departement.csv")
        
        # Prétraitement dates
        for df in [df_france, df_regions, df_departements]:
            if '1er jour de la semaine' in df.columns:
                df['Date'] = pd.to_datetime(df['1er jour de la semaine'], errors='coerce')
                df['Année'] = df['Date'].dt.year
                df['Mois'] = df['Date'].dt.month
                df['Semaine_ISO'] = df['Date'].dt.isocalendar().week
                df['Trimestre'] = df['Date'].dt.quarter
        
        return {
            'france': df_france,
            'regions': df_regions,
            'departements': df_departements,
            'vacc_france': df_vacc_france,
            'vacc_regions': df_vacc_regions,
            'vacc_depts': df_vacc_depts
        }
    
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {e}")
        return None

@st.cache_data
def compute_master_dataset(data_dict):
    """Crée le dataset maître avec KPIs"""
    
    df_urg = data_dict['departements'].copy()
    df_vacc = data_dict['vacc_depts'].copy()
    
    # Agrégation urgences
    df_urg_agg = df_urg.groupby(['Département Code', 'Département', 'Région']).agg({
        'Taux de passages aux urgences pour grippe': 'mean',
        'Taux d\'hospitalisations après passages aux urgences pour grippe': 'mean',
        'Taux d\'actes médicaux SOS médecins pour grippe': 'mean'
    }).reset_index()
    
    df_urg_agg.columns = ['Code_Dept', 'Département', 'Région', 
                           'Taux_Urgences_Moyen', 'Taux_Hospit_Moyen', 'Taux_SOS_Moyen']
    
    # Vaccination récente
    annee_max = df_vacc['Année'].max()
    df_vacc_recent = df_vacc[df_vacc['Année'] == annee_max].copy()
    
@st.cache_data
def compute_master_dataset(data_dict):
    """Crée le dataset maître avec KPIs"""
    
    df_urg = data_dict['departements'].copy()
    df_vacc = data_dict['vacc_depts'].copy()
    
    # Agrégation urgences
    df_urg_agg = df_urg.groupby(['Département Code', 'Département', 'Région']).agg({
        'Taux de passages aux urgences pour grippe': 'mean',
        'Taux d\'hospitalisations après passages aux urgences pour grippe': 'mean',
        'Taux d\'actes médicaux SOS médecins pour grippe': 'mean'
    }).reset_index()
    
    df_urg_agg.columns = ['Code_Dept', 'Département', 'Région', 
                           'Taux_Urgences_Moyen', 'Taux_Hospit_Moyen', 'Taux_SOS_Moyen']
    
    # Vaccination récente
    annee_max = df_vacc['Année'].max()
    df_vacc_recent = df_vacc[df_vacc['Année'] == annee_max].copy()
    
    # Fusion
    df_master = df_urg_agg.merge(
        df_vacc_recent[['Département Code', 'Grippe 65 ans et plus', 
                        'Grippe 65-74 ans', 'Grippe 75 ans et plus', 'Année']],
        left_on='Code_Dept',
        right_on='Département Code',
        how='left'
    )
    
    df_master.rename(columns={
        'Grippe 65 ans et plus': 'Couverture_65plus',
        'Grippe 65-74 ans': 'Couverture_65_74',
        'Grippe 75 ans et plus': 'Couverture_75plus'
    }, inplace=True)
    
    # === DIAGNOSTIC ET IMPUTATION AMÉLIORÉE ===
    nb_avant = len(df_master)
    print(f"\n   📊 Diagnostic NaN après fusion : {nb_avant} départements")
    
    for col in ['Couverture_65plus', 'Couverture_65_74', 'Couverture_75plus']:
        nb_nan = df_master[col].isna().sum()
        if nb_nan > 0:
            mediane = df_master[col].median()
            print(f"   ⚠️  {nb_nan} NaN dans {col} → Imputation par médiane ({mediane:.1f}%)")
            df_master[col].fillna(mediane, inplace=True)
        else:
            print(f"   ✓ {col} : aucun NaN")
    
    # Vérifier que l'imputation a fonctionné
    assert df_master['Couverture_65plus'].isna().sum() == 0, "Erreur : NaN restants dans Couverture_65plus"
    
    # Calcul KPIs (sécurisé)
    df_master['Score_Impact'] = (
        df_master['Taux_Urgences_Moyen'] * 
        (100 - df_master['Couverture_65plus']) / 10
    ).round(1)
    
    moyenne_nationale = df_master['Couverture_65plus'].mean()
    df_master['Gap_Vaccinal'] = (moyenne_nationale - df_master['Couverture_65plus']).round(1)
    
    coef_calibre = -0.65
    df_master['Potentiel_Reduction_Urgences'] = (
        df_master['Gap_Vaccinal'] * coef_calibre
    ).abs().round(1)
    
    urgences_norm = (df_master['Taux_Urgences_Moyen'] / df_master['Taux_Urgences_Moyen'].max()) * 100
    gap_norm = ((100 - df_master['Couverture_65plus']) / 50) * 100
    hospit_norm = (df_master['Taux_Hospit_Moyen'] / df_master['Taux_Hospit_Moyen'].max()) * 100
    
    df_master['Indice_Vulnerabilite'] = (
        urgences_norm * 0.4 +
        gap_norm.clip(0, 100) * 0.3 +
        hospit_norm * 0.3
    ).round(1)
    
    df_master['Priorité_Action'] = (
        df_master['Score_Impact'] * 0.5 +
        df_master['Gap_Vaccinal'] * 10 +
        df_master['Taux_Hospit_Moyen'] * 25
    ).round(0)
    
    df_master['Catégorie_Risque'] = pd.cut(
        df_master['Score_Impact'],
        bins=[0, 250, 500, 750, float('inf')],
        labels=['Faible', 'Moyen', 'Élevé', 'Critique']
    )
    
    df_master['Population_65plus_Estimee'] = 100000 * 0.20
    df_master['Doses_Necessaires'] = (
        df_master['Population_65plus_Estimee'] * 
        df_master['Gap_Vaccinal'] / 100
    ).round(0)
    
    # === NETTOYAGE FINAL ===
    # Supprimer les lignes avec NaN dans colonnes critiques pour ML
    colonnes_critiques = ['Taux_Urgences_Moyen', 'Taux_Hospit_Moyen', 'Score_Impact', 
                          'Couverture_65plus', 'Gap_Vaccinal', 'Doses_Necessaires']
    
    nb_nan_final = df_master[colonnes_critiques].isna().any(axis=1).sum()
    if nb_nan_final > 0:
        print(f"   ⚠️  {nb_nan_final} départements avec NaN résiduels → Suppression")
        df_master = df_master.dropna(subset=colonnes_critiques)
    
    print(f"   ✓ Dataset final : {len(df_master)} départements propres\n")
    
    df_master = df_master.sort_values('Priorité_Action', ascending=False).reset_index(drop=True)
    df_master['Année_Référence'] = annee_max
    
    return df_master  # ← FERMETURE DE LA FONCTION


# Fonction normalisation codes départements
def normaliser_code_dept(code):
    """Normalise les codes départements pour la carte"""
    if pd.isna(code):
        return None
    code_str = str(code).strip()
    if code_str.isdigit():
        return code_str.zfill(2)
    return code_str

# Fonction normalisation codes départements
def normaliser_code_dept(code):
    """Normalise les codes départements pour la carte"""
    if pd.isna(code):
        return None
    code_str = str(code).strip()
    if code_str.isdigit():
        return code_str.zfill(2)
    return code_str

# =============================================================================
# CHARGEMENT DONNÉES
# =============================================================================

data = load_all_data()
if data is None:
    st.error("❌ Impossible de charger les données")
    st.stop()

df_master = compute_master_dataset(data)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("## 🎯 Navigation")

pages = {
    "🏠 Tableau de Bord": "dashboard",
    "🗺️ Cartographie": "map",
    "📈 Prédictions ML": "predictions",
    "🎯 Simulateur": "simulator",
    "📥 Export": "export"
}

page = st.sidebar.radio("Choisissez une page :", list(pages.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Filtres Globaux")

annees_disponibles = sorted(data['departements']['Année'].dropna().unique())
if len(annees_disponibles) > 0:
    annee_selectionnee = st.sidebar.selectbox("📅 Année", ['Toutes'] + [int(a) for a in annees_disponibles], index=0)
else:
    annee_selectionnee = 'Toutes'

regions_disponibles = ['Toutes'] + sorted(df_master['Région'].unique().tolist())
region_filter = st.sidebar.selectbox("📍 Région", regions_disponibles)

risque_filter = st.sidebar.multiselect(
    "⚠️ Niveau de risque",
    ['Critique', 'Élevé', 'Moyen', 'Faible'],
    default=['Critique', 'Élevé']
)

# Appliquer filtres
df_filtered = df_master.copy()
if region_filter != 'Toutes':
    df_filtered = df_filtered[df_filtered['Région'] == region_filter]
if risque_filter:
    df_filtered = df_filtered[df_filtered['Catégorie_Risque'].isin(risque_filter)]

st.sidebar.markdown("---")
st.sidebar.info(f"""
**📊 Données filtrées**
- {len(df_filtered)} départements
- Année : {annee_selectionnee}
- Région : {region_filter}
""")

# =============================================================================
# PAGE 1 : TABLEAU DE BORD
# =============================================================================

if pages[page] == "dashboard":
    st.markdown('<div class="main-header">🏠 Tableau de Bord Stratégique</div>', unsafe_allow_html=True)
    
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        couv_moyenne = df_master['Couverture_65plus'].mean()
        delta_couv = couv_moyenne - df_master['Couverture_65plus'].quantile(0.25)
        st.metric("💉 Couverture Moyenne 65+", f"{couv_moyenne:.1f}%", f"+{delta_couv:.1f}% vs Q1")
    
    with col2:
        urgences_tot = df_master['Taux_Urgences_Moyen'].sum()
        st.metric("🏥 Passages Urgences", f"{urgences_tot:,.0f}", "cumul/100k hab.")
    
    with col3:
        dept_critiques = (df_master['Catégorie_Risque'] == 'Critique').sum()
        pct_critiques = dept_critiques / len(df_master) * 100
        st.metric("🚨 Dép. Critiques", dept_critiques, f"{pct_critiques:.1f}%", delta_color="inverse")
    
    with col4:
        potentiel_total = df_master['Potentiel_Reduction_Urgences'].sum()
        st.metric("📉 Potentiel Réduction", f"{potentiel_total:,.0f}", "urgences/an")
    
    with col5:
        doses_totales = df_master['Doses_Necessaires'].sum()
        st.metric("💉 Doses Nécessaires", f"{doses_totales/1000:.0f}k", "objectif 75%")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Évolution", "💉 Couverture", "🏥 Urgences", "🎯 Départements", "🌡️ Heatmap"
    ])
    
    with tab1:
        st.markdown("### 📈 Évolution Temporelle")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            indicateur = st.selectbox("Indicateur", [
                'Taux de passages aux urgences pour grippe',
                'Taux d\'hospitalisations après passages aux urgences pour grippe',
                'Taux d\'actes médicaux SOS médecins pour grippe'
            ])
        
        with col2:
            echelle = st.selectbox("Échelle", ['National', 'Régional', 'Départemental'])
        
        if echelle == 'National':
            df_plot = data['france'].copy()
            df_plot = df_plot[df_plot['Date'].notna()].sort_values('Date')
            
            fig = px.line(df_plot, x='Date', y=indicateur, 
                         color='Classe d\'âge' if 'Classe d\'âge' in df_plot.columns else None,
                         title=f"Évolution Nationale - {indicateur}")
            fig.update_layout(height=500, hovermode='x unified')
            st.plotly_chart(fig, key="evolution_national", width="stretch")
        
        elif echelle == 'Régional':
            df_plot = data['regions'].copy()
            df_plot = df_plot[df_plot['Date'].notna()].sort_values('Date')
            regions = st.multiselect("Régions", sorted(df_plot['Région'].unique()), 
                                    default=sorted(df_plot['Région'].unique())[:5])
            
            if regions:
                df_plot_filtered = df_plot[df_plot['Région'].isin(regions)]
                fig = px.line(df_plot_filtered, x='Date', y=indicateur, color='Région',
                             title=f"Évolution Régionale - {indicateur}")
                fig.update_layout(height=500, hovermode='x unified')
                st.plotly_chart(fig, key="evolution_regional", width="stretch")
        
        else:
            top10 = df_master.head(10)['Département'].tolist()
            df_plot = data['departements'].copy()
            df_plot = df_plot[df_plot['Date'].notna()].sort_values('Date')
            df_plot = df_plot[df_plot['Département'].isin(top10)]
            
            if not df_plot.empty:
                fig = px.line(df_plot, x='Date', y=indicateur, color='Département',
                             title=f"Évolution Top 10 - {indicateur}")
                fig.update_layout(height=500, hovermode='x unified')
                st.plotly_chart(fig, key="evolution_departemental", width="stretch")
    
    with tab2:
        st.markdown("### 💉 Couverture Vaccinale")
        col1, col2 = st.columns(2)
        
        with col1:
            df_regions_vacc = df_master.groupby('Région')['Couverture_65plus'].mean().reset_index()
            df_regions_vacc = df_regions_vacc.sort_values('Couverture_65plus', ascending=False)
            
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                y=df_regions_vacc['Région'],
                x=df_regions_vacc['Couverture_65plus'],
                orientation='h',
                marker_color='#2ecc71',
                text=df_regions_vacc['Couverture_65plus'].round(1),
                textposition='outside'
            ))
            fig1.update_layout(title="Couverture 65+ par Région", 
                              xaxis_title="Couverture (%)", height=600)
            fig1.add_vline(x=75, line_dash="dash", line_color="red", annotation_text="Objectif 75%")
            fig1.update_yaxes(autorange="reversed")
            st.plotly_chart(fig1, key="couverture_region", width="stretch")
        
        with col2:
            df_vacc_temps = data['vacc_france'].copy()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_vacc_temps['Année'], 
                                     y=df_vacc_temps['Grippe 65 ans et plus'],
                                     mode='lines+markers', name='65+ ans',
                                     line=dict(color='#2ecc71', width=3)))
            fig2.update_layout(title="Évolution Couverture (2011-2024)", height=600)
            st.plotly_chart(fig2, key="couverture_evolution", width="stretch")
    
    with tab3:
        st.markdown("### 🏥 Urgences & Hospitalisations")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df_master, x='Couverture_65plus', y='Taux_Urgences_Moyen',
                           size='Score_Impact', color='Catégorie_Risque',
                           color_discrete_map={'Faible':'#28a745', 'Moyen':'#ffc107',
                                              'Élevé':'#fd7e14', 'Critique':'#dc3545'},
                           hover_data=['Département', 'Région'],
                           title="Corrélation Couverture vs Urgences")
            
            # Ligne de tendance
            X = df_master[['Couverture_65plus']].values
            y = df_master['Taux_Urgences_Moyen'].values
            model = LinearRegression()
            model.fit(X, y)
            x_trend = np.linspace(X.min(), X.max(), 100)
            y_trend = model.predict(x_trend.reshape(-1, 1))
            
            fig.add_trace(go.Scatter(x=x_trend.flatten(), y=y_trend, mode='lines',
                                    name='Tendance', line=dict(dash='dash', color='red', width=2)))
            fig.update_layout(height=500)
            st.plotly_chart(fig, key="scatter_couverture_urgences", width="stretch")
        
        with col2:
            fig = px.box(df_master, x='Catégorie_Risque', y='Taux_Hospit_Moyen',
                        color='Catégorie_Risque',
                        color_discrete_map={'Faible':'#28a745', 'Moyen':'#ffc107',
                                           'Élevé':'#fd7e14', 'Critique':'#dc3545'},
                        title="Taux Hospitalisation par Risque")
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, key="boxplot_hospit_risque", width="stretch")
    
    with tab4:
        st.markdown("### 🎯 Départements Prioritaires")
        metrique = st.selectbox("Classer par", ['Priorité_Action', 'Score_Impact', 
                                                 'Gap_Vaccinal', 'Taux_Urgences_Moyen'])
        
        df_top20 = df_filtered.sort_values(metrique, ascending=False).head(20)
        
        colors_map = {'Critique':'#dc3545', 'Élevé':'#fd7e14', 
                     'Moyen':'#ffc107', 'Faible':'#28a745'}
        colors = [colors_map.get(cat, '#6c757d') for cat in df_top20['Catégorie_Risque']]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(y=df_top20['Département'], x=df_top20[metrique],
                            orientation='h', marker_color=colors,
                            text=df_top20[metrique].round(1), textposition='outside'))
        fig.update_layout(title=f"Top 20 - {metrique}", height=700)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, key="top20_departements", width="stretch")
        
        st.dataframe(df_top20[['Département', 'Région', 'Catégorie_Risque',
                               'Couverture_65plus', 'Gap_Vaccinal', 'Score_Impact']], 
                    use_container_width=True, height=400)
    
    with tab5:
        st.markdown("### 🌡️ Heatmap Régionale")
        
        df_hm = df_master.groupby('Région').agg({
            'Taux_Urgences_Moyen': 'mean',
            'Couverture_65plus': 'mean',
            'Score_Impact': 'mean',
            'Gap_Vaccinal': 'mean',
            'Taux_Hospit_Moyen': 'mean'
        }).reset_index()
        
        # Normaliser
        for col in ['Taux_Urgences_Moyen', 'Score_Impact', 'Taux_Hospit_Moyen', 'Gap_Vaccinal']:
            df_hm[f'{col}_norm'] = ((df_hm[col] - df_hm[col].min()) / 
                                   (df_hm[col].max() - df_hm[col].min()) * 100)
        
        z_data = df_hm[['Taux_Urgences_Moyen_norm', 'Gap_Vaccinal_norm', 
                        'Score_Impact_norm', 'Taux_Hospit_Moyen_norm']].T.values
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=df_hm['Région'],
            y=['Taux Urgences', 'Gap Vaccinal', 'Score Impact', 'Taux Hospit.'],
            colorscale='RdYlGn_r',
            text=df_hm[['Taux_Urgences_Moyen', 'Gap_Vaccinal', 
                        'Score_Impact', 'Taux_Hospit_Moyen']].T.values.round(1),
            texttemplate='%{text}',
            textfont=dict(size=9)
        ))
        fig.update_layout(title="Heatmap Régionale", height=500)
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, key="heatmap_regionale", width="stretch")

# =============================================================================
# PAGE 2 : CARTOGRAPHIE
# =============================================================================

elif pages[page] == "map":
    st.markdown('<div class="main-header">🗺️ Cartographie Intelligente</div>', unsafe_allow_html=True)
    
    st.markdown("### Carte Interactive de France")
    
    # Sélection indicateur
    indicateur_carte = st.selectbox("Indicateur à afficher", [
        'Score_Impact', 'Priorité_Action', 'Couverture_65plus', 
        'Gap_Vaccinal', 'Taux_Urgences_Moyen', 'Indice_Vulnerabilite'
    ])
    
    labels = {
        'Score_Impact': "Score d'Impact",
        'Priorité_Action': "Priorité d'Action",
        'Couverture_65plus': "Couverture 65+ (%)",
        'Gap_Vaccinal': "Gap Vaccinal (pts)",
        'Taux_Urgences_Moyen': "Taux Urgences",
        'Indice_Vulnerabilite': "Indice Vulnérabilité"
    }
    
    # Normaliser codes
    df_map = df_master.copy()
    df_map['Code_Dept_Clean'] = df_map['Code_Dept'].apply(normaliser_code_dept)
    
    # Télécharger GeoJSON
    try:
        import urllib.request
        geojson_url = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
        with urllib.request.urlopen(geojson_url) as url:
            departements_geojson = json.loads(url.read().decode())
        
        # Colorscale
        if indicateur_carte in ['Couverture_65plus']:
            colorscale = 'RdYlGn'
        else:
            colorscale = 'RdYlGn_r'
        
        fig = go.Figure(go.Choroplethmapbox(
            geojson=departements_geojson,
            locations=df_map['Code_Dept_Clean'],
            z=df_map[indicateur_carte],
            featureidkey="properties.code",
            colorscale=colorscale,
            marker_opacity=0.7,
            marker_line_width=1,
            marker_line_color='white',
            colorbar=dict(title=labels[indicateur_carte]),
            text=df_map['Département'],
            hovertemplate='<b>%{text}</b><br>' + 
                         f'{labels[indicateur_carte]}: %{{z:.1f}}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=4.8,
            mapbox_center={"lat": 46.8, "lon": 2.5},
            height=700,
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, key="carte_france", width="stretch")
        
        # Stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_dept = df_map.loc[df_map[indicateur_carte].idxmax()]
            st.error(f"""
            **🔴 Maximum**
            
            {top_dept['Département']}
            
            {labels[indicateur_carte]}: **{top_dept[indicateur_carte]:.1f}**
            """)
        
        with col2:
            bottom_dept = df_map.loc[df_map[indicateur_carte].idxmin()]
            st.success(f"""
            **🟢 Minimum**
            
            {bottom_dept['Département']}
            
            {labels[indicateur_carte]}: **{bottom_dept[indicateur_carte]:.1f}**
            """)
        
        with col3:
            mean_val = df_map[indicateur_carte].mean()
            st.warning(f"""
            **📊 Moyenne**
            
            Nationale
            
            {labels[indicateur_carte]}: **{mean_val:.1f}**
            """)
    
    except Exception as e:
        st.error(f"Erreur carte : {e}")
        st.info("Affichage alternatif : Top 10 départements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔝 Top 10")
            top10 = df_map.nlargest(10, indicateur_carte)
            fig = go.Figure(data=[go.Bar(y=top10['Département'], x=top10[indicateur_carte],
                                        orientation='h', marker_color='#dc3545')])
            fig.update_layout(height=400)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, key="carte_fallback_top10", width="stretch")
        
        with col2:
            st.markdown("### 🔻 Bottom 10")
            bottom10 = df_map.nsmallest(10, indicateur_carte)
            fig = go.Figure(data=[go.Bar(y=bottom10['Département'], x=bottom10[indicateur_carte],
                                        orientation='h', marker_color='#28a745')])
            fig.update_layout(height=400)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, key="carte_fallback_bottom10", width="stretch")

# =============================================================================
# PAGE 3 : PRÉDICTIONS PROPHET
# =============================================================================

if pages[page] == "predictions":
    st.markdown('<div class="main-header">📈 Prédictions avec Prophet</div>', unsafe_allow_html=True)
    
    # === VÉRIFICATION MODÈLES PROPHET ===
    @st.cache_resource
    def load_prophet_models():
        """Charge les modèles Prophet pré-entraînés"""
        from pathlib import Path
        import joblib
        
        models_dir = Path("models/prophet")
        
        if not models_dir.exists():
            return None, None, None
        
        try:
            model_nat = joblib.load(models_dir / "model_national.pkl")
            data_nat = joblib.load(models_dir / "data_national.pkl")
            metadata = joblib.load(models_dir / "metadata.pkl")
            
            # Charge départements disponibles
            models_dept = {}
            for dept in metadata.get('departements', []):
                dept_safe = dept.replace(' ', '_').replace('-', '_')
                model_path = models_dir / f"model_{dept_safe}.pkl"
                if model_path.exists():
                    models_dept[dept] = joblib.load(model_path)
            
            return model_nat, data_nat, models_dept
        
        except Exception as e:
            st.error(f"❌ Erreur chargement modèles : {e}")
            return None, None, None
    
    # === CHARGEMENT ===
    model_national, data_national, models_dept = load_prophet_models()
    
    if model_national is None:
        st.warning("""
        ⚠️ **Modèles Prophet non disponibles**
        
        Les modèles Prophet doivent être entraînés au préalable.
        
        **Actions requises :**
        1. Exécute `python scripts/simulation/train_prophet_models.py`
        2. Attends la fin de l'entraînement (~2-5 min)
        3. Recharge cette page
        """)
        
        # Fallback : Affiche graphiques existants
        st.markdown("---")
        st.markdown("### 📊 Analyse Historique (en attendant Prophet)")
        
        df_nat = data['france'].copy()
        df_nat = df_nat[df_nat['Date'].notna()].sort_values('Date')
        
        fig = px.line(
            df_nat, 
            x='Date', 
            y='Taux de passages aux urgences pour grippe',
            title="Historique National - Passages aux Urgences"
        )
        fig.update_layout(height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        st.stop()  # Arrête l'exécution si modèles absents
    
    # === SI MODÈLES DISPONIBLES : INTERFACE PRINCIPALE ===
    
    st.markdown(f"""
    **Modèles entraînés** : {len(models_dept)} départements  
    **Données** : {len(data_national)} semaines d'historique
    """)
    
    
    vacc_multiplier = {
    '-10%': 0.90, 
    '-5%': 0.95, 
    'Default': 1.0,
    '+5%': 1.05, 
    '+10%': 1.10, 
    '+15%': 1.15
}
    # === TABS ===
    tab1, tab2, tab3 = st.tabs([
        "📈 National", 
        "🗺️ Départements", 
        "🎯 Scénarios Vaccination"
    ])
    
    # =========================================================================
    # TAB 1 : PRÉDICTIONS NATIONALES
    # =========================================================================
    with tab1:
        st.markdown("### 📈 Prédictions Nationales (France)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            periods = st.slider("Semaines à prédire", 12, 104, 52, 4)
        
        with col2:
            scenario_vacc = st.select_slider(
                "Scénario vaccination",
                options=['-10%', '-5%', 'Default', '+5%', '+10%', '+15%'],
                value='Default'
            )
        
        # === GÉNÉRATION PRÉDICTIONS ===
        if st.button("🚀 Générer Prédictions", key="btn_pred_nat"):
            
            with st.spinner("🔄 Calcul en cours..."):
                # Crée le futur
                future = model_national.make_future_dataframe(periods=periods, freq='W')
                
                # Applique scénario vaccination
                last_vacc_rate = data_national['vaccination_rate'].iloc[-1]
                vacc_multiplier = {
                    '-10%': 0.90, '-5%': 0.95, 'Default': 1.0,
                    '+5%': 1.05, '+10%': 1.10, '+15%': 1.15
                }
                future['vaccination_rate'] = last_vacc_rate * vacc_multiplier[scenario_vacc]
                
                # Prédit
                forecast = model_national.predict(future)
                
                # Stocke en session
                st.session_state['forecast_nat'] = forecast
                st.session_state['scenario_nat'] = scenario_vacc
        
        # === AFFICHAGE RÉSULTATS ===
        if 'forecast_nat' in st.session_state:
            forecast = st.session_state['forecast_nat']
            
            # Métriques
            st.markdown("#### 📊 Métriques de Performance")
            
            # Calcul MAE sur historique
            historical_forecast = forecast[forecast['ds'].isin(data_national['ds'])]
            data_aligned = data_national.merge(historical_forecast[['ds', 'yhat']], on='ds')
            mae = np.mean(np.abs(data_aligned['y'] - data_aligned['yhat']))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE (Erreur Absolue)", f"{mae:.2f}%")
            col2.metric("Semaines prédites", f"{periods}")
            col3.metric("Scénario", st.session_state['scenario_nat'])
            
            # Graphique principal
            st.markdown("#### 🎯 Prédictions vs Historique")
            
            fig = go.Figure()
            
            # Historique
            fig.add_trace(go.Scatter(
                x=data_national['ds'], 
                y=data_national['y'],
                mode='markers',
                name='Données réelles',
                marker=dict(color='#1f77b4', size=5, opacity=0.6)
            ))
            
            # Prédictions
            fig.add_trace(go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat'],
                mode='lines',
                name='Prédictions',
                line=dict(color='#ff7f0e', width=3)
            ))
            
            # Intervalle de confiance
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalle 95%',
                showlegend=True
            ))
            
            today = data_national['ds'].max()
            
            fig.add_shape(
                type="line",
                x0=today, x1=today,  # Même valeur pour ligne verticale
                y0=0, y1=1,
                yref="paper",  # Coordonnées relatives (0=bas, 1=haut)
                line=dict(color="gray", width=2, dash="dash")
            )
            
            fig.add_annotation(
                x=today,
                y=1.02,  # Légèrement au-dessus du graphique
                yref="paper",
                text="Aujourd'hui",
                showarrow=False,
                font=dict(size=12, color="gray"),
                xanchor="center"
            )
            
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Taux passages urgences (%)',
                hovermode='x unified',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prédiction prochain pic
            st.markdown("#### 🔥 Prochain Pic Épidémique")
            
            future_data = forecast[forecast['ds'] > today]
            peak_date = future_data.loc[future_data['yhat'].idxmax(), 'ds']
            peak_value = future_data['yhat'].max()
            peak_upper = future_data.loc[future_data['yhat'].idxmax(), 'yhat_upper']
            peak_lower = future_data.loc[future_data['yhat'].idxmax(), 'yhat_lower']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📅 Date du pic", peak_date.strftime('%d/%m/%Y'))
            col2.metric("📈 Intensité prévue", f"{peak_value:.1f}%")
            col3.metric("📊 Fourchette", f"{peak_lower:.1f}% - {peak_upper:.1f}%")
            
            # Alerte
            mean_historical = data_national['y'].mean()
            if peak_value > mean_historical * 1.5:
                st.error(f"""
                ⚠️ **ALERTE** : Le pic prévu est **{(peak_value/mean_historical - 1)*100:.0f}% supérieur** 
                à la moyenne historique ({mean_historical:.1f}%)
                """)
            else:
                st.success(f"""
                ✅ Pic modéré : {(peak_value/mean_historical - 1)*100:.0f}% au-dessus de la moyenne
                """)
            
            # Export
            st.download_button(
                "📥 Télécharger les prédictions (CSV)",
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8'),
                f"predictions_prophet_national_{scenario_vacc}.csv",
                "text/csv"
            )
    
    # =========================================================================
    # TAB 2 : PRÉDICTIONS DÉPARTEMENTALES
    # =========================================================================
    with tab2:
        st.markdown("### 🗺️ Prédictions par Département")
        
        if not models_dept:
            st.warning("⚠️ Aucun modèle départemental disponible")
        else:
            # Filtre départements disponibles
            depts_disponibles = list(models_dept.keys())
            
            # Appliquer filtres globaux
            if region_filter != 'Toutes':
                depts_region = df_master[df_master['Région'] == region_filter]['Département'].tolist()
                depts_disponibles = [d for d in depts_disponibles if d in depts_region]
            
            if not depts_disponibles:
                st.warning("Aucun département disponible avec les filtres actuels")
            else:
                dept_choisi = st.selectbox("Département", depts_disponibles)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    periods_dept = st.slider("Semaines à prédire", 12, 104, 52, 4, key="periods_dept")
                with col2:
                    scenario_vacc_dept = st.select_slider(
                        "Scénario vaccination",
                        options=['-10%', '-5%', 'Default', '+5%', '+10%', '+15%'],
                        value='Default',
                        key="scenario_dept"
                    )
                
                if st.button("🚀 Générer Prédictions", key="btn_pred_dept"):
                    
                    with st.spinner(f"🔄 Calcul pour {dept_choisi}..."):
                        import joblib
                        
                        # Charge modèle et données
                        dept_safe = dept_choisi.replace(' ', '_').replace('-', '_')
                        model_dept = models_dept[dept_choisi]
                        data_dept = joblib.load(f"models/prophet/data_{dept_safe}.pkl")
                        
                        # Prédictions
                        future_dept = model_dept.make_future_dataframe(periods=periods_dept, freq='W')
                        last_vacc_dept = data_dept['vaccination_rate'].iloc[-1]
                        future_dept['vaccination_rate'] = last_vacc_dept * vacc_multiplier[scenario_vacc_dept]
                        
                        forecast_dept = model_dept.predict(future_dept)
                        
                        st.session_state['forecast_dept'] = forecast_dept
                        st.session_state['data_dept'] = data_dept
                        st.session_state['dept_name'] = dept_choisi
                
                # Affichage
                if 'forecast_dept' in st.session_state and st.session_state['dept_name'] == dept_choisi:
                    forecast_dept = st.session_state['forecast_dept']
                    data_dept = st.session_state['data_dept']
                    
                    # Graphique
                    fig_dept = go.Figure()
                    
                    fig_dept.add_trace(go.Scatter(
                        x=data_dept['ds'], y=data_dept['y'],
                        mode='markers', name='Réel',
                        marker=dict(color='#1f77b4', size=4)
                    ))
                    
                    fig_dept.add_trace(go.Scatter(
                        x=forecast_dept['ds'], y=forecast_dept['yhat'],
                        mode='lines', name='Prédiction',
                        line=dict(color='#ff7f0e', width=2)
                    ))
                    
                    fig_dept.add_trace(go.Scatter(
                        x=forecast_dept['ds'].tolist() + forecast_dept['ds'].tolist()[::-1],
                        y=forecast_dept['yhat_upper'].tolist() + forecast_dept['yhat_lower'].tolist()[::-1],
                        fill='toself', fillcolor='rgba(255, 127, 14, 0.15)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Intervalle 95%'
                    ))
                    
                    today_dept = data_dept['ds'].max()

                    fig_dept.add_shape(
                        type="line",
                        x0=today_dept, x1=today_dept,
                        y0=0, y1=1,
                        yref="paper",
                        line=dict(color="gray", width=2, dash="dash")
                    )

                    fig_dept.add_annotation(
                        x=today_dept,
                        y=1.02,
                        yref="paper",
                        text="Aujourd'hui",
                        showarrow=False,
                        font=dict(size=11, color="gray")
                    )
                    
                    fig_dept.update_layout(
                        title=f"Prédictions pour {dept_choisi}",
                        xaxis_title='Date',
                        yaxis_title='Taux urgences (%)',
                        height=600
                    )
                    
                    st.plotly_chart(fig_dept, use_container_width=True)
    
    # === GUIDE D'INTERPRÉTATION ===
    with st.expander("📖 Comment interpréter les prédictions Prophet ?"):
        st.markdown("""
        ### Éléments du graphique
        - **Points bleus** : Données historiques réelles (2011-2024)
        - **Ligne orange** : Prédictions du modèle Prophet
        - **Zone orange claire** : Intervalle de confiance à 95%
        
        ### Scénarios de vaccination
        - **Default** : Maintien du taux actuel (~50%)
        - **+10%** : Augmentation à 55% (campagne ciblée)
        - **+15%** : Augmentation à 57.5% (campagne ambitieuse)
        
        ### Limites du modèle
        ⚠️ Prophet suppose que les patterns historiques se répètent  
        ⚠️ Ne prend PAS en compte : nouveaux variants, changements climatiques  
        ⚠️ L'incertitude augmente avec l'horizon temporel
        
        ### Méthodologie
        - **Modèle** : Prophet (Facebook AI Research)
        - **Saisonnalité** : Multiplicative (adapté aux épidémies)
        - **Variables** : Tendance + Saison + Couverture vaccinale
        - **Entraînement** : 14 ans de données (2011-2024)
        """)

# =============================================================================
# PAGE 4 : SIMULATEUR
# =============================================================================

if pages[page] == "simulator":
    st.markdown('<div class="main-header">🎯 Simulateur Enrichi</div>', unsafe_allow_html=True)
    
    st.markdown("### 🏥 Simulateur d'Impact des Actions de Vaccination")
    
    # Sélection département
    dept_selectionne = st.selectbox("📍 Département", df_master['Département'].unique())
    dept_info = df_master[df_master['Département'] == dept_selectionne].iloc[0]
    
    # Baseline
    st.markdown("---")
    st.markdown("### 📊 État Actuel (Baseline)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💉 Couverture", f"{dept_info['Couverture_65plus']:.1f}%")
    
    with col2:
        st.metric("🏥 Taux Urgences", f"{dept_info['Taux_Urgences_Moyen']:.1f}")
    
    with col3:
        st.metric("📊 Gap Vaccinal", f"{dept_info['Gap_Vaccinal']:.1f} pts")
    
    with col4:
        badge_class = {
            'Critique': 'badge-critique',
            'Élevé': 'badge-eleve',
            'Moyen': 'badge-moyen',
            'Faible': 'badge-faible'
        }.get(dept_info['Catégorie_Risque'], 'badge-moyen')
        
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem 0;'>
            <span class='{badge_class}'>{dept_info['Catégorie_Risque']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Configuration actions
    st.markdown("### ⚙️ Configuration des Actions")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💉 Doses Vaccins", "🏪 Pharmacies", 
                                       "🚑 SOS Médecins", "📣 Communication"])
    
    actions = {}
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            nb_doses = st.number_input("Nombre de doses", 0, 100000, 10000, 1000)
            tranche_age = st.selectbox("Tranche d'âge", ['65-74 ans', '75+ ans', 'Tous 65+'])
    with col2:
        # Plage étendue : semaine 1 à 52 (toute l'année)
        semaine = st.slider("Semaine implémentation", 1, 52, 40)

        # Efficacité selon période
        # Pic épidémique : semaines 48-10 (novembre-mars) → 80%
        # Pré-pic : semaines 36-47 (septembre-novembre) → 70%
        # Hors saison : reste de l'année → 50%
        if (semaine >= 48 or semaine <= 10):
            efficacite = 0.80
            periode = "Pic épidémique"
        elif 36 <= semaine <= 47:
            efficacite = 0.70
            periode = "Pré-pic (optimal)"
        else:
            efficacite = 0.50
            periode = "Hors saison"

        st.info(f"**Efficacité : {efficacite*100:.0f}%**\n\n{periode}")

        actions['doses'] = {'actif': nb_doses > 0, 'valeur': nb_doses,
                           'efficacite': efficacite, 'cout': nb_doses * 12}
    
    with tab2:
        nb_pharmacies = st.number_input("Nombre de pharmacies", 0, 20, 3, 1)
        impact_pharmacie = 0.09 * nb_pharmacies
        st.metric("Impact sur couverture", f"+{impact_pharmacie:.1f}%")
        actions['pharmacies'] = {'actif': nb_pharmacies > 0, 'valeur': nb_pharmacies,
                                'impact': impact_pharmacie, 'cout': nb_pharmacies * 2500}
    
    with tab3:
        nb_sos = st.number_input("Nombre d'équipes SOS", 0, 10, 2, 1)
        impact_sos = 0.06 * nb_sos
        st.metric("Impact sur couverture", f"+{impact_sos:.1f}%")
        actions['sos'] = {'actif': nb_sos > 0, 'valeur': nb_sos,
                         'impact': impact_sos, 'cout': nb_sos * 80000}
    
    with tab4:
        budget_comm = st.number_input("Budget (milliers €)", 0, 500, 50, 10)
        impact_comm = 0.04 * (budget_comm / 100)
        st.metric("Impact sur couverture", f"+{impact_comm:.1f}%")
        actions['comm'] = {'actif': budget_comm > 0, 'valeur': budget_comm,
                          'impact': impact_comm, 'cout': budget_comm * 1000}
    
    st.markdown("---")
    
    # Lancer simulation
    if st.button("🚀 LANCER LA SIMULATION", type="primary", use_container_width=True):
        
        with st.spinner("⏳ Calcul en cours..."):
            # Calcul impact cumulé
            delta_couverture = 0
            
            # Doses
            if actions['doses']['actif']:
                pop_estimee = dept_info['Population_65plus_Estimee']
                delta_couverture += (actions['doses']['valeur'] / pop_estimee) * 100 * actions['doses']['efficacite']
            
            # Pharmacies
            if actions['pharmacies']['actif']:
                delta_couverture += actions['pharmacies']['impact']
            
            # SOS
            if actions['sos']['actif']:
                delta_couverture += actions['sos']['impact']
            
            # Communication
            if actions['comm']['actif']:
                delta_couverture += actions['comm']['impact']
            
            # Cap à +15 pts
            delta_couverture = min(delta_couverture, 15)
            
            # Impact urgences (coefficient calibré)
            coef_urgences = -0.65
            delta_urgences = delta_couverture * coef_urgences
            
            # Impact hospitalisations
            delta_hospit = delta_urgences * (dept_info['Taux_Hospit_Moyen'] / 100)
            
            # Simulation
            simulation = {
                'couverture': dept_info['Couverture_65plus'] + delta_couverture,
                'urgences': dept_info['Taux_Urgences_Moyen'] + delta_urgences,
                'hospitalisations': dept_info['Taux_Hospit_Moyen'] + delta_hospit
            }
            
            # Bénéfices
            cout_passage = 190
            cout_hospit = 3800
            
            urgences_evitees = abs(delta_urgences) * 52  # Par an
            hospit_evitees = abs(delta_hospit) * 52
            
            benefice_urgences = urgences_evitees * cout_passage
            benefice_hospit = hospit_evitees * cout_hospit
            benefice_total = benefice_urgences + benefice_hospit
            
            # Coûts
            cout_total = sum(a['cout'] for a in actions.values() if a['actif'])
            
            # ROI
            roi = ((benefice_total - cout_total) / cout_total * 100) if cout_total > 0 else 0
        
        st.success("✅ Simulation terminée !")
        
        # Résultats
        st.markdown("---")
        st.markdown("### 📊 Résultats de la Simulation")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💉 Couverture 65+", f"{simulation['couverture']:.1f}%", 
                     f"+{delta_couverture:.1f} pts")
        
        with col2:
            st.metric("🏥 Taux Urgences", f"{simulation['urgences']:.1f}", 
                     f"{delta_urgences:+.1f}")
        
        with col3:
            st.metric("🚑 Urgences Évitées/an", f"{urgences_evitees:.0f}")
        
        with col4:
            roi_color = "normal" if roi > 0 else "inverse"
            st.metric("💰 ROI", f"{roi:+.0f}%", delta_color=roi_color)
        
        # Graphique comparaison
        st.markdown("### 📈 Comparaison Avant / Après")
        
        categories = ['Couverture 65+', 'Taux Urgences']
        avant = [dept_info['Couverture_65plus'], dept_info['Taux_Urgences_Moyen']]
        apres = [simulation['couverture'], simulation['urgences']]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Avant', x=categories, y=avant, 
                            marker_color='#ff7f0e', text=avant, textposition='auto'))
        fig.add_trace(go.Bar(name='Après', x=categories, y=apres, 
                            marker_color='#2ca02c', text=[round(v,1) for v in apres], 
                            textposition='auto'))
        
        fig.update_layout(barmode='group', height=400, title="Impact de la Simulation")
        st.plotly_chart(fig, key="sim_avant_apres", width="stretch")
        
        # Analyse financière
        st.markdown("### 💰 Analyse Financière")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💸 Coûts")
            couts_data = []
            if actions['doses']['actif']:
                couts_data.append(('Doses vaccins', actions['doses']['cout']))
            if actions['pharmacies']['actif']:
                couts_data.append(('Pharmacies', actions['pharmacies']['cout']))
            if actions['sos']['actif']:
                couts_data.append(('SOS Médecins', actions['sos']['cout']))
            if actions['comm']['actif']:
                couts_data.append(('Communication', actions['comm']['cout']))
            
            if couts_data:
                df_couts = pd.DataFrame(couts_data, columns=['Poste', 'Montant'])
                fig = px.pie(df_couts, values='Montant', names='Poste', 
                            title="Répartition des Coûts", hole=0.3)
                fig.update_layout(height=300)
                st.plotly_chart(fig, key="sim_couts_camembert", width="stretch")
            
            st.metric("💵 Coût Total", f"{cout_total:,.0f} €")
        
        with col2:
            st.markdown("#### 💎 Bénéfices")
            benefices_data = [
                ('Économies Urgences', benefice_urgences),
                ('Économies Hospitalisations', benefice_hospit)
            ]
            df_benefices = pd.DataFrame(benefices_data, columns=['Poste', 'Montant'])
            fig = px.pie(df_benefices, values='Montant', names='Poste',
                        title="Répartition des Bénéfices", hole=0.3)
            fig.update_layout(height=300)
            st.plotly_chart(fig, key="sim_benefices_camembert", width="stretch")
            
            st.metric("💚 Bénéfice Total", f"{benefice_total:,.0f} €")
        
        # Interprétation
        st.markdown("---")
        st.markdown("### 💡 Interprétation")
        
        if roi > 200:
            st.success(f"""
            🎉 **EXCELLENT ROI ({roi:.0f}%) !**
            
            Cette stratégie est très rentable. Pour chaque euro investi, 
            vous économisez **{benefice_total/cout_total:.2f} €**.
            
            ✅ Déployer immédiatement
            """)
        elif roi > 50:
            st.info(f"""
            👍 **BON ROI ({roi:.0f}%)**
            
            La stratégie est rentable.
            
            💡 Peut être optimisée pour encore plus d'impact
            """)
        elif roi > 0:
            st.warning(f"""
            ⚠️ **ROI FAIBLE ({roi:.0f}%)**
            
            Rentable mais nécessite optimisation.
            
            🔧 Ajuster les paramètres ou cibler mieux
            """)
        else:
            st.error(f"""
            ❌ **ROI NÉGATIF ({roi:.0f}%)**
            
            Coût > Bénéfices
            
            🛑 NE PAS déployer en l'état
            """)

# =============================================================================
# PAGE 5 : EXPORT
# =============================================================================

if pages[page] == "export":
    st.markdown('<div class="main-header">📥 Export & Rapports</div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Télécharger les Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_master = df_master.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Télécharger Dataset Maître (CSV)",
            data=csv_master,
            file_name=f"master_dataset_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col2:
        df_critiques = df_master[df_master['Catégorie_Risque'] == 'Critique']
        csv_critiques = df_critiques.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="🚨 Télécharger Départements Critiques (CSV)",
            data=csv_critiques,
            file_name=f"departements_critiques_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques d'Export")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📂 Lignes Dataset", len(df_master))
    
    with col2:
        st.metric("🚨 Départements Critiques", len(df_critiques))
    
    with col3:
        st.metric("📊 Colonnes", len(df_master.columns))
    
    with col4:
        taille_mo = len(csv_master) / 1024 / 1024
        st.metric("💾 Taille", f"{taille_mo:.2f} MB")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem 0;">
    <p style="font-size: 1.1rem; font-weight: bold;">🦠 Hackathon Stratégie Vaccinale Grippe 💉</p>
    <p>Dashboard Complet - 5 Pages Fonctionnelles</p>
    <p style="font-size: 0.9rem; color: #999;">
        Données : Santé Publique France | Année : 2011 - 2024
    </p>
</div>
""", unsafe_allow_html=True)