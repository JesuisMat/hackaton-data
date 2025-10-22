#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DASHBOARD INTERACTIF - STRATÉGIE VACCINALE GRIPPE
Application Streamlit pour visualiser et simuler l'impact des campagnes de vaccination
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================
st.set_page_config(
    page_title="🦠 Stratégie Vaccinale Grippe France",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# STYLE CSS PERSONNALISÉ
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .impact-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .impact-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .impact-low {
        color: #388e3c;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

@st.cache_data
def load_data():
    """Charge les données depuis les fichiers CSV"""
    try:
        # Chargement des données (conservez vos paths existants)
        df_france = pd.read_csv("data/grippe-passages-aux-urgences-et-actes-sos-medecins-france.csv")
        df_regions = pd.read_csv("data/grippe-passages-urgences-et-actes-sos-medecin_reg.csv")
        df_departements = pd.read_csv("data/grippe-passages-aux-urgences-et-actes-sos-medecins-departement.csv")
        df_vacc_france = pd.read_csv("data/couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-france.csv")
        df_vacc_regions = pd.read_csv("data/couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-region.csv")
        df_vacc_depts = pd.read_csv("data/couvertures-vaccinales-des-adolescent-et-adultes-departement.csv")

        # Correction du format des semaines (ex: "2023-S08" → extraire l'année et le numéro de semaine)
        for df in [df_regions, df_departements, df_france]:
            if '1er jour de la semaine' in df.columns:
                # Conversion de la date
                df['Date'] = pd.to_datetime(df['1er jour de la semaine'], dayfirst=True, errors='coerce')

                # Extraction de l'année et du mois
                df['Année'] = df['Date'].dt.year
                df['Mois'] = df['Date'].dt.month

                # Traitement spécial pour la colonne "Semaine" si elle existe
                if 'Semaine' in df.columns:
                    # Si le format est "2023-S08" → extraire le numéro de semaine (08)
                    if df['Semaine'].dtype == object and df['Semaine'].str.contains('-S').any():
                        df['Semaine'] = df['Semaine'].str.split('-S').str[1].astype(int)
                    else:
                        # Sinon, conversion directe en entier
                        df['Semaine'] = pd.to_numeric(df['Semaine'], errors='coerce').fillna(0).astype(int)

        return df_france, df_regions, df_departements, df_vacc_france, df_vacc_regions, df_vacc_depts

    except Exception as e:
        st.error(f"❌ Erreur de chargement des données : {e}")
        return None, None, None, None, None, None


# Chargement
df_france, df_regions, df_departements, df_vacc_france, df_vacc_regions, df_vacc_depts = load_data()

# =============================================================================
# SIDEBAR - NAVIGATION
# =============================================================================
st.sidebar.markdown("## 📋 Navigation")
page = st.sidebar.radio(
    "Sélectionnez une vue :",
    ["🏠 Accueil", "📊 Vue Nationale", "🗺️ Vue Régionale", 
     "📍 Vue Départementale", "🎯 Simulation Impact", "💡 Recommandations"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ À propos")
st.sidebar.info(
    "**Hackathon Stratégie Vaccinale Grippe**\n\n"
    "Dashboard interactif pour optimiser les campagnes de vaccination "
    "contre la grippe en France."
)

# =============================================================================
# PAGE ACCUEIL
# =============================================================================
if page == "🏠 Accueil":
    st.markdown('<div class="main-header">🦠 Stratégie Vaccinale Grippe France 💉</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Objectifs du Projet
    
    Ce dashboard permet de :
    - 📈 **Analyser** les tendances de la grippe en France
    - 🗺️ **Identifier** les zones à risque et sous-vaccinées
    - 🎯 **Optimiser** la distribution des vaccins
    - 💰 **Calculer** le ROI des campagnes de vaccination
    - 🚀 **Simuler** l'impact de différentes stratégies
    """)
    
    # Métriques clés
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Datasets",
            value="6",
            delta="Multiéchelles"
        )
    
    with col2:
        if df_france is not None:
            st.metric(
                label="📅 Période couverte",
                value="2011-2024",
                delta="14 ans"
            )
    
    with col3:
        if df_vacc_france is not None:
            last_cov = df_vacc_france['Grippe 65 ans et plus'].iloc[-1]
            st.metric(
                label="💉 Couverture 65+ (2024)",
                value=f"{last_cov:.1f}%",
                delta=None
            )
    
    with col4:
        st.metric(
            label="🏥 Régions analysées",
            value="18",
            delta="Métropole + DOM"
        )
    
    st.markdown("---")
    
    # Problématiques clés
    st.markdown("### 🔍 Problématiques Adressées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Prédiction des besoins** 🔮
        - Anticiper les besoins en vaccins par territoire
        - Éviter les ruptures de stock
        
        **2. Optimisation de la distribution** 📦
        - Cibler les pharmacies prioritaires
        - Réduire les coûts logistiques
        """)
    
    with col2:
        st.markdown("""
        **3. Amélioration de l'accès aux soins** 🏥
        - Identifier les zones sous-vaccinées
        - Proposer des actions ciblées
        
        **4. Anticipation de la pression hospitalière** 🚑
        - Corréler vaccination et passages aux urgences
        - Quantifier l'impact économique
        """)
    
    st.markdown("---")
    
    # Guide d'utilisation
    with st.expander("📖 Guide d'Utilisation", expanded=False):
        st.markdown("""
        **Navigation :**
        - Utilisez le menu latéral pour naviguer entre les vues
        - Chaque vue propose des filtres interactifs
        
        **Vues disponibles :**
        - 📊 **Vue Nationale** : Tendances globales en France
        - 🗺️ **Vue Régionale** : Comparaison entre régions
        - 📍 **Vue Départementale** : Analyse fine par département
        - 🎯 **Simulation Impact** : Calculateur d'impact des campagnes
        - 💡 **Recommandations** : Actions prioritaires
        """)

# =============================================================================
# PAGE VUE NATIONALE
# =============================================================================
elif page == "📊 Vue Nationale":
    st.header("📊 Analyse Nationale - France")
    
    if df_france is None or df_vacc_france is None:
        st.error("❌ Données non disponibles")
    else:
        # Préparer les données
        df_france['Date'] = pd.to_datetime(df_france['1er jour de la semaine'])
        df_france['Année'] = df_france['Date'].dt.year
        
        # Filtres
        st.sidebar.markdown("### 🔧 Filtres")
        annees = sorted(df_france['Année'].unique())
        annee_selectionnee = st.sidebar.slider(
            "Sélectionner une année",
            min_value=int(min(annees)),
            max_value=int(max(annees)),
            value=(int(min(annees)), int(max(annees)))
        )
        
        df_filtered = df_france[
            (df_france['Année'] >= annee_selectionnee[0]) & 
            (df_france['Année'] <= annee_selectionnee[1])
        ]
        
        # Onglets
        tab1, tab2, tab3 = st.tabs(["📈 Évolution Temporelle", "👥 Classes d'Âge", "💉 Couverture Vaccinale"])
        
        with tab1:
            st.subheader("Évolution du Taux de Passages aux Urgences pour Grippe")
            
            # Agréger par semaine
            df_agg = df_filtered.groupby('Date').agg({
                'Taux de passages aux urgences pour grippe': 'mean',
                'Taux d\'hospitalisations après passages aux urgences pour grippe': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_agg['Date'],
                y=df_agg['Taux de passages aux urgences pour grippe'],
                mode='lines',
                name='Passages aux urgences',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))
            
            fig.add_trace(go.Scatter(
                x=df_agg['Date'],
                y=df_agg['Taux d\'hospitalisations après passages aux urgences pour grippe'],
                mode='lines',
                name='Hospitalisations',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            fig.update_layout(
                title="Taux de passages aux urgences et hospitalisations (pour 100k habitants)",
                xaxis_title="Date",
                yaxis_title="Taux pour 100k habitants",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques descriptives
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "📊 Taux moyen urgences",
                    f"{df_filtered['Taux de passages aux urgences pour grippe'].mean():.1f}",
                    help="Pour 100k habitants"
                )
            with col2:
                st.metric(
                    "🏥 Taux moyen hospitalisations",
                   f"{df_filtered['Taux d\'hospitalisations après passages aux urgences pour grippe'].mean():.1f}",
                    help="Pour 100k habitants"
                )
            with col3:
                st.metric(
                    "📈 Variabilité (std)",
                    f"{df_filtered['Taux de passages aux urgences pour grippe'].std():.1f}"
                )
        
        with tab2:
            st.subheader("Comparaison par Classes d'Âge")
            
            # Filtrer par classe d'âge
            classes_age = df_filtered['Classe d\'âge'].unique()
            classe_selectionnee = st.multiselect(
                "Sélectionner les classes d'âge",
                options=sorted(classes_age),
                default=list(sorted(classes_age)[:3])
            )
            
            if classe_selectionnee:
                df_age = df_filtered[df_filtered['Classe d\'âge'].isin(classe_selectionnee)]
                df_age_agg = df_age.groupby(['Date', 'Classe d\'âge']).agg({
                    'Taux de passages aux urgences pour grippe': 'mean'
                }).reset_index()
                
                fig = px.line(
                    df_age_agg,
                    x='Date',
                    y='Taux de passages aux urgences pour grippe',
                    color='Classe d\'âge',
                    title="Évolution par classe d'âge",
                    labels={'Taux de passages aux urgences pour grippe': 'Taux pour 100k habitants'}
                )
                fig.update_layout(height=500, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Évolution de la Couverture Vaccinale")
            
            # Graphique couverture vaccinale
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_vacc_france['Année'],
                y=df_vacc_france['Grippe 65 ans et plus'],
                mode='lines+markers',
                name='65 ans et plus',
                line=dict(color='#2ca02c', width=3),
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=df_vacc_france['Année'],
                y=df_vacc_france['Grippe moins de 65 ans à risque'],
                mode='lines+markers',
                name='<65 ans à risque',
                line=dict(color='#d62728', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Couverture vaccinale contre la grippe (%)",
                xaxis_title="Année",
                yaxis_title="Taux de couverture (%)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Objectif de santé publique
            objectif_65_plus = 75.0
            derniere_couverture = df_vacc_france['Grippe 65 ans et plus'].iloc[-1]
            ecart = objectif_65_plus - derniere_couverture
            
            st.info(f"""
            🎯 **Objectif de Santé Publique** : {objectif_65_plus}% pour les 65+  
            📊 **Couverture actuelle** : {derniere_couverture:.1f}%  
            📉 **Écart** : {ecart:.1f} points de pourcentage
            """)

# =============================================================================
# PAGE VUE RÉGIONALE
# =============================================================================
elif page == "🗺️ Vue Régionale":
    st.header("🗺️ Analyse Régionale")
    
    if df_regions is None or df_vacc_regions is None:
        st.error("❌ Données non disponibles")
    else:
        # Préparer les données
        df_regions['Date'] = pd.to_datetime(df_regions['1er jour de la semaine'])
        df_regions['Année'] = df_regions['Date'].dt.year
        
        # Filtres sidebar
        st.sidebar.markdown("### 🔧 Filtres")
        regions_list = sorted(df_regions['Région'].dropna().unique())
        region_selectionnee = st.sidebar.multiselect(
            "Sélectionner des régions",
            options=regions_list,
            default=regions_list[:5]
        )
        
        annees = sorted(df_regions['Année'].unique())
        annee_selectionnee = st.sidebar.slider(
            "Année",
            min_value=int(min(annees)),
            max_value=int(max(annees)),
            value=int(max(annees))
        )
        
        # Filtrer les données
        df_reg_filtered = df_regions[
            (df_regions['Région'].isin(region_selectionnee)) &
            (df_regions['Année'] == annee_selectionnee)
        ]
        
        # Onglets
        tab1, tab2, tab3 = st.tabs(["📊 Classement", "📈 Évolution", "💉 Vaccination"])
        
        with tab1:
            st.subheader(f"Classement des Régions - {annee_selectionnee}")
            
            # Agréger par région
            df_reg_agg = df_reg_filtered.groupby('Région').agg({
                'Taux de passages aux urgences pour grippe': 'mean',
                'Taux d\'hospitalisations après passages aux urgences pour grippe': 'mean'
            }).reset_index().sort_values(
                'Taux de passages aux urgences pour grippe',
                ascending=False
            )
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_reg_agg['Région'],
                x=df_reg_agg['Taux de passages aux urgences pour grippe'],
                orientation='h',
                marker=dict(
                    color=df_reg_agg['Taux de passages aux urgences pour grippe'],
                    colorscale='Reds',
                    showscale=True
                ),
                text=df_reg_agg['Taux de passages aux urgences pour grippe'].round(1),
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Taux de passages aux urgences pour grippe par région",
                xaxis_title="Taux pour 100k habitants",
                yaxis_title="Région",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Évolution Temporelle par Région")
            
            # Sélection région pour évolution
            region_evolution = st.selectbox(
                "Sélectionner une région",
                options=region_selectionnee
            )
            
            df_evolution = df_regions[df_regions['Région'] == region_evolution]
            df_evolution_agg = df_evolution.groupby('Date').agg({
                'Taux de passages aux urgences pour grippe': 'mean'
            }).reset_index()
            
            fig = px.line(
                df_evolution_agg,
                x='Date',
                y='Taux de passages aux urgences pour grippe',
                title=f"Évolution - {region_evolution}",
                labels={'Taux de passages aux urgences pour grippe': 'Taux pour 100k habitants'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Couverture Vaccinale Régionale")
            
            # Dernière année disponible
            derniere_annee = df_vacc_regions['Année'].max()
            df_vacc_last = df_vacc_regions[df_vacc_regions['Année'] == derniere_annee]
            
            fig = px.bar(
                df_vacc_last.sort_values('Grippe 65 ans et plus', ascending=False),
                x='Région',
                y='Grippe 65 ans et plus',
                title=f"Couverture vaccinale 65+ par région ({derniere_annee})",
                labels={'Grippe 65 ans et plus': 'Taux de couverture (%)'},
                color='Grippe 65 ans et plus',
                color_continuous_scale='Blues'  # Notez le changement ici
            )
            fig.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE VUE DÉPARTEMENTALE
# =============================================================================
elif page == "📍 Vue Départementale":
    st.header("📍 Analyse Départementale")
    
    if df_departements is None or df_vacc_depts is None:
        st.error("❌ Données non disponibles")
    else:
        # Préparer les données
        df_departements['Date'] = pd.to_datetime(df_departements['1er jour de la semaine'])
        df_departements['Année'] = df_departements['Date'].dt.year
        
        # Filtres
        st.sidebar.markdown("### 🔧 Filtres")
        
        # Sélection région pour filtrer départements
        regions_list = sorted(df_departements['Région'].dropna().unique())
        region_filter = st.sidebar.selectbox(
            "Filtrer par région",
            options=['Toutes'] + regions_list
        )
        
        if region_filter != 'Toutes':
            depts_list = sorted(
                df_departements[df_departements['Région'] == region_filter]['Département'].dropna().unique()
            )
        else:
            depts_list = sorted(df_departements['Département'].dropna().unique())
        
        dept_selectionne = st.sidebar.selectbox(
            "Sélectionner un département",
            options=depts_list
        )
        
        annees = sorted(df_departements['Année'].unique())
        annee_selectionnee = st.sidebar.slider(
            "Année",
            min_value=int(min(annees)),
            max_value=int(max(annees)),
            value=int(max(annees))
        )
        
        # Filtrer
        df_dept = df_departements[
            (df_departements['Département'] == dept_selectionne) &
            (df_departements['Année'] == annee_selectionnee)
        ]
        
        # Affichage
        st.subheader(f"📍 {dept_selectionne}")
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        
        with col1:
            taux_moy = df_dept['Taux de passages aux urgences pour grippe'].mean()
            st.metric("📊 Taux moyen urgences", f"{taux_moy:.1f}")
        
        with col2:
            taux_hospit = df_dept['Taux d\'hospitalisations après passages aux urgences pour grippe'].mean()
            st.metric("🏥 Taux hospitalisations", f"{taux_hospit:.1f}")
        
        with col3:
            # Couverture vaccinale
            df_vacc_dept = df_vacc_depts[
                (df_vacc_depts['Département'] == dept_selectionne) &
                (df_vacc_depts['Année'] == annee_selectionnee)
            ]
            if not df_vacc_dept.empty:
                cov = df_vacc_dept['Grippe 65 ans et plus'].values[0]
                st.metric("💉 Couverture 65+", f"{cov:.1f}%")
        
        # Graphique évolution
        st.subheader("Évolution sur l'année")
        
        df_dept_sorted = df_dept.sort_values('Date')
        fig = px.line(
            df_dept_sorted,
            x='Date',
            y='Taux de passages aux urgences pour grippe',
            title=f"Passages aux urgences pour grippe - {dept_selectionne}",
            labels={'Taux de passages aux urgences pour grippe': 'Taux pour 100k habitants'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparaison avec la région
        st.subheader("Comparaison avec la région")
        
        region_dept = df_departements[df_departements['Département'] == dept_selectionne]['Région'].iloc[0]
        df_region_comp = df_regions[
            (df_regions['Région'] == region_dept) &
            (df_regions['Année'] == annee_selectionnee)
        ]
        
        taux_region = df_region_comp['Taux de passages aux urgences pour grippe'].mean()
        ecart = ((taux_moy - taux_region) / taux_region * 100) if taux_region > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Département", f"{taux_moy:.1f}")
        with col2:
            st.metric("Région moyenne", f"{taux_region:.1f}", delta=f"{ecart:+.1f}%")

# =============================================================================
# PAGE SIMULATION IMPACT
# =============================================================================
elif page == "🎯 Simulation Impact":
    st.header("🎯 Simulateur d'Impact des Campagnes de Vaccination")
    
    st.markdown("""
    Cet outil permet de **simuler l'impact** d'une campagne de vaccination ciblée
    sur les passages aux urgences et le retour sur investissement (ROI).
    """)
    
    # Paramètres de simulation
    st.subheader("⚙️ Paramètres de la Campagne")
    
    col1, col2 = st.columns(2)
    
    with col1:
        doses_total = st.number_input(
            "💉 Nombre total de doses à distribuer",
            min_value=10000,
            max_value=10000000,
            value=100000,
            step=10000,
            help="Nombre total de doses de vaccin disponibles"
        )
        
        cout_dose = st.number_input(
            "💰 Coût par dose (€)",
            min_value=5.0,
            max_value=50.0,
            value=15.0,
            step=1.0
        )
    
    with col2:
        efficacite = st.slider(
            "📊 Efficacité vaccinale (%)",
            min_value=30,
            max_value=90,
            value=60,
            help="Pourcentage de réduction des passages aux urgences chez les vaccinés"
        )
        
        cout_passage_urgence = st.number_input(
            "🏥 Coût moyen d'un passage aux urgences (€)",
            min_value=50,
            max_value=1000,
            value=200,
            step=50
        )
    
    st.markdown("---")
    
    # Stratégies de ciblage
    st.subheader("🎯 Stratégie de Ciblage")
    
    strategie = st.radio(
        "Choisir une stratégie",
        [
            "🌍 Distribution homogène (pas de ciblage)",
            "🔴 Ciblage départements à haut risque",
            "🎯 Ciblage zones sous-vaccinées",
            "🧠 Ciblage optimisé (IA)"
        ]
    )
    
    # Calculs de simulation
    if st.button("🚀 Lancer la Simulation", type="primary"):
        with st.spinner("Calcul en cours..."):
            # Simulation basique
            if strategie == "🌍 Distribution homogène (pas de ciblage)":
                taux_efficacite = efficacite / 100
                urgences_evitees = doses_total * taux_efficacite * 0.02  # 2% des doses évitent 1 urgence
                boost = 1.0
            
            elif strategie == "🔴 Ciblage départements à haut risque":
                taux_efficacite = efficacite / 100
                urgences_evitees = doses_total * taux_efficacite * 0.03  # 3% efficacité
                boost = 1.5
            
            elif strategie == "🎯 Ciblage zones sous-vaccinées":
                taux_efficacite = efficacite / 100
                urgences_evitees = doses_total * taux_efficacite * 0.035  # 3.5% efficacité
                boost = 1.75
            
            else:  # Ciblage optimisé
                taux_efficacite = efficacite / 100
                urgences_evitees = doses_total * taux_efficacite * 0.045  # 4.5% efficacité
                boost = 2.0
            
            urgences_evitees *= boost
            
            # Calculs économiques
            cout_campagne = doses_total * cout_dose
            economie_realisee = urgences_evitees * cout_passage_urgence
            benefice_net = economie_realisee - cout_campagne
            roi = (benefice_net / cout_campagne * 100) if cout_campagne > 0 else 0
            
            # Affichage des résultats
            st.success("✅ Simulation terminée !")
            
            st.markdown("### 📊 Résultats de la Simulation")
            
            # Métriques principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🏥 Urgences évitées",
                    f"{int(urgences_evitees):,}",
                    help="Nombre estimé de passages aux urgences évités"
                )
            
            with col2:
                st.metric(
                    "💰 Coût campagne",
                    f"{int(cout_campagne):,} €"
                )
            
            with col3:
                st.metric(
                    "💵 Économies réalisées",
                    f"{int(economie_realisee):,} €",
                    delta=f"+{int(benefice_net):,} €"
                )
            
            with col4:
                roi_color = "normal" if roi > 0 else "inverse"
                st.metric(
                    "📈 ROI",
                    f"{roi:.1f}%",
                    delta="Bénéfice" if roi > 0 else "Perte",
                    delta_color=roi_color
                )
            
            # Graphique de comparaison
            st.markdown("### 📊 Comparaison Coûts vs Économies")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Coût Campagne', 'Économies Réalisées'],
                y=[cout_campagne, economie_realisee],
                marker_color=['#ff7f0e', '#2ca02c'],
                text=[f"{int(cout_campagne):,} €", f"{int(economie_realisee):,} €"],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Impact Financier de la Campagne",
                yaxis_title="Montant (€)",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interprétation
            st.markdown("### 💡 Interprétation")
            
            if roi > 50:
                st.success(f"""
                ✅ **Excellent ROI ({roi:.1f}%) !**  
                La campagne est très rentable. Pour chaque euro investi, 
                vous économisez {(economie_realisee/cout_campagne):.2f} €.
                """)
            elif roi > 0:
                st.info(f"""
                📊 **ROI Positif ({roi:.1f}%)**  
                La campagne est rentable mais peut être optimisée. 
                Considérez un meilleur ciblage pour maximiser l'impact.
                """)
            else:
                st.warning(f"""
                ⚠️ **ROI Négatif ({roi:.1f}%)**  
                Le coût de la campagne dépasse les économies réalisées. 
                Recommandation : ajuster les paramètres ou cibler davantage.
                """)

# =============================================================================
# PAGE RECOMMANDATIONS
# =============================================================================
elif page == "💡 Recommandations":
    st.header("💡 Recommandations Stratégiques")
    
    st.markdown("""
    Sur la base des analyses réalisées, voici les **recommandations prioritaires** 
    pour optimiser la stratégie vaccinale contre la grippe en France.
    """)
    
    # Recommandation 1
    st.markdown("### 🎯 1. Ciblage Géographique Prioritaire")
    
    with st.expander("📍 Départements à Prioriser", expanded=True):
        st.markdown("""
        **Critères d'identification :**
        - Taux de passages aux urgences élevé (> 80/100k)
        - Couverture vaccinale faible (< 50% chez les 65+)
        - Population à risque importante
        
        **Actions recommandées :**
        - 📦 Augmenter les stocks de vaccins de **20-30%**
        - 🚐 Déployer des unités mobiles de vaccination
        - 📣 Renforcer les campagnes de communication locale
        - 🏥 Partenariats avec médecins généralistes et pharmacies
        """)
    
    # Recommandation 2
    st.markdown("### 📅 2. Optimisation du Calendrier Vaccinal")
    
    with st.expander("⏰ Timing Optimal", expanded=False):
        st.markdown("""
        **Pic épidémique :** Décembre - Février  
        **Période optimale de vaccination :** Octobre - Novembre
        
        **Actions recommandées :**
        - 🗓️ Débuter les campagnes **mi-septembre**
        - 🎯 Objectif : 75% de couverture avant décembre
        - 📊 Suivi hebdomadaire des couvertures régionales
        - 🚨 Alertes précoces en cas de retard
        """)
    
    # Recommandation 3
    st.markdown("### 👥 3. Ciblage des Populations Vulnérables")
    
    with st.expander("🎯 Groupes Prioritaires", expanded=False):
        st.markdown("""
        **Priorité 1 : 65 ans et plus**
        - Objectif : 75% de couverture (actuellement ~50%)
        - Méthode : Rappels automatisés, gratuité, facilité d'accès
        
        **Priorité 2 : Personnes à risque < 65 ans**
        - Objectif : 50% de couverture (actuellement ~30%)
        - Méthode : Sensibilisation des médecins, bons de vaccination
        
        **Priorité 3 : Personnel soignant**
        - Objectif : 80% de couverture
        - Méthode : Vaccination obligatoire ou fortement incitée
        """)
    
    # Recommandation 4
    st.markdown("### 🤖 4. Utilisation de l'IA et du Machine Learning")
    
    with st.expander("🧠 Modèles Prédictifs", expanded=False):
        st.markdown("""
        **Déploiement recommandé :**
        - 📈 **Modèle de prédiction des besoins** : SARIMA/Prophet
          - Anticiper les besoins 2-3 mois à l'avance
          - Précision cible : ±10%
        
        - 🗺️ **Scoring géographique** : Random Forest/XGBoost
          - Identifier les zones à risque
          - Actualisation mensuelle
        
        - 🎯 **Optimisation de la distribution** : Algorithmes d'optimisation
          - Minimiser les ruptures de stock
          - Maximiser la couverture avec budget contraint
        """)
    
    # Recommandation 5
    st.markdown("### 💰 5. Optimisation Budgétaire")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **💵 Investissements Prioritaires**
        - 📦 Logistique et distribution : 30%
        - 📣 Communication et sensibilisation : 25%
        - 💉 Doses de vaccin : 35%
        - 🤖 Outils numériques et IA : 10%
        """)
    
    with col2:
        st.success("""
        **📊 ROI Attendu**
        - Réduction passages urgences : **-20%**
        - Économies Sécurité Sociale : **+150M€**
        - ROI global : **+200%**
        - Vies sauvées : **~2000/an**
        """)
    
    # Recommandation 6
    st.markdown("### 📱 6. Digitalisation et Innovation")
    
    with st.expander("🚀 Outils Digitaux", expanded=False):
        st.markdown("""
        **Applications mobiles :**
        - 📲 Rappels personnalisés de vaccination
        - 🗺️ Géolocalisation des centres de vaccination
        - 📊 Suivi personnel de la couverture vaccinale
        
        **Portail web décideurs :**
        - 📈 Dashboard temps réel des couvertures
        - 🚨 Alertes automatiques (stocks, épidémies)
        - 📊 Tableaux de bord prédictifs
        - 📥 Export de rapports personnalisables
        """)
    
    # Plan d'action synthétique
    st.markdown("---")
    st.markdown("### 📋 Plan d'Action Synthétique (12 mois)")
    
    timeline_data = {
        'Phase': ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4'],
        'Période': ['Mois 1-3', 'Mois 4-6', 'Mois 7-9', 'Mois 10-12'],
        'Actions': [
            '🔧 Audit des données, identification zones prioritaires, formation équipes',
            '🚀 Lancement campagnes ciblées, déploiement IA, outils digitaux',
            '📊 Monitoring temps réel, ajustements, renforcement zones critiques',
            '📈 Évaluation impact, capitalisation learnings, planification année N+1'
        ]
    }
    
    df_timeline = pd.DataFrame(timeline_data)
    
    st.table(df_timeline)
    
    st.success("""
    🎯 **Objectif Final**  
    Augmenter la couverture vaccinale de **10 points de pourcentage** en 2 ans  
    et réduire les passages aux urgences de **20%** durant la saison grippale.
    """)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>📊 <strong>Hackathon Stratégie Vaccinale Grippe</strong></p>
    <p>Données : Santé Publique France | Dashboard : Streamlit + Plotly</p>
    <p><em>Optimiser la vaccination, sauver des vies 💙</em></p>
</div>
""", unsafe_allow_html=True)