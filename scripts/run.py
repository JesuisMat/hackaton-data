#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT PRINCIPAL - LANCEMENT COMPLET DE L'EDA
Lance séquentiellement les 3 analyses : Nationale → Régionale → Départementale
Durée totale estimée : 20-25 minutes
"""

import subprocess
import sys
import os
from datetime import datetime

print("="*90)
print(" HACKATHON STRATÉGIE VACCINALE GRIPPE - ANALYSE EXPLORATOIRE COMPLÈTE ".center(90, "="))
print("="*90)
print(f"\n🕐 Démarrage : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print(f"🐍 Python : {sys.version.split()[0]}\n")

# =============================================================================
# VÉRIFICATION STRUCTURE DU PROJET
# =============================================================================
print("📁 Vérification de la structure du projet...")
print("-"*90)

# Détection du répertoire de travail
if os.path.exists('../data'):
    BASE_DIR = '..'
    SCRIPTS_DIR = '.'
elif os.path.exists('./data'):
    BASE_DIR = '.'
    SCRIPTS_DIR = './scripts'
else:
    print("❌ ERREUR : Structure de projet non reconnue")
    print("   Assurez-vous d'avoir les dossiers 'data' et 'scripts'")
    sys.exit(1)

DATA_DIR = f"{BASE_DIR}/data"
OUTPUT_DIR = f"{BASE_DIR}/output"

# Création du dossier output
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fichiers requis
fichiers_requis = {
    'National': 'grippe-passages-aux-urgences-et-actes-sos-medecins-france.csv',
    'Régional': 'grippe-passages-urgences-et-actes-sos-medecin_reg.csv',
    'Départemental': 'grippe-passages-aux-urgences-et-actes-sos-medecins-departement.csv'
}

print(f"✓ Répertoire data : {DATA_DIR}")
print(f"✓ Répertoire output : {OUTPUT_DIR}")
print(f"\n📋 Vérification des fichiers CSV :")

tous_presents = True
for nom, fichier in fichiers_requis.items():
    chemin = f"{DATA_DIR}/{fichier}"
    if os.path.exists(chemin):
        taille = os.path.getsize(chemin) / (1024*1024)
        print(f"   ✓ {nom:15s} : {fichier:50s} ({taille:5.1f} MB)")
    else:
        print(f"   ❌ {nom:15s} : MANQUANT - {fichier}")
        tous_presents = False

if not tous_presents:
    print("\n❌ Certains fichiers sont manquants. Impossible de continuer.")
    sys.exit(1)

print("\n✅ Tous les fichiers sont présents !")

# =============================================================================
# PLAN D'EXÉCUTION
# =============================================================================
print("\n" + "="*90)
print(" PLAN D'EXÉCUTION ".center(90, "="))
print("="*90)

scripts = [
    {
        'nom': 'ANALYSE NATIONALE',
        'fichier': 'exploreNat.py',
        'duree': '~5 min',
        'outputs': ['01_evolution_nationale.png', '02_saisonnalite.png', '03_comparaison_classes_age.png']
    },
    {
        'nom': 'ANALYSE RÉGIONALE',
        'fichier': 'exploreReg.py',
        'duree': '~8 min',
        'outputs': ['04_classement_regions.png', '05_evolution_regions.png', '06_hospitalisations_regions.png']
    },
    {
        'nom': 'ANALYSE DÉPARTEMENTALE',
        'fichier': 'exploreDpt.py',
        'duree': '~10 min',
        'outputs': ['07_top20_departements.png', '08_variabilite_regionale.png', '09_evolution_top_departements.png']
    }
]

for i, script in enumerate(scripts, 1):
    print(f"\n{i}. {script['nom']}")
    print(f"   Script : {script['fichier']}")
    print(f"   Durée : {script['duree']}")
    print(f"   Outputs : {len(script['outputs'])} graphiques")

print(f"\n⏱️  DURÉE TOTALE ESTIMÉE : ~25 minutes")
print("="*90)

# Demande de confirmation
print("\n" + "⚠️ " * 20)
reponse = input("\n▶️  Lancer l'analyse complète ? (o/n) : ").strip().lower()
print()

if reponse != 'o':
    print("❌ Analyse annulée par l'utilisateur")
    sys.exit(0)

# =============================================================================
# EXÉCUTION SÉQUENTIELLE
# =============================================================================
print("="*90)
print(" EXÉCUTION DES ANALYSES ".center(90, "="))
print("="*90)

resultats = []

for i, script in enumerate(scripts, 1):
    print(f"\n{'='*90}")
    print(f" [{i}/3] {script['nom']} ".center(90, "="))
    print(f"{'='*90}\n")
    
    script_path = f"{SCRIPTS_DIR}/{script['fichier']}"
    
    if not os.path.exists(script_path):
        print(f"⚠️ ATTENTION : {script['fichier']} introuvable dans {SCRIPTS_DIR}/")
        print(f"   Recherche du script à la racine...")
        script_path = f"./{script['fichier']}"
        
        if not os.path.exists(script_path):
            print(f"❌ Script introuvable. Passage à l'analyse suivante.")
            resultats.append({'nom': script['nom'], 'statut': 'SKIP', 'duree': 0})
            continue
    
    debut = datetime.now()
    
    try:
        print(f"▶️  Exécution de {script['fichier']}...\n")
        
        # Exécution du script Python
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True
        )
        
        fin = datetime.now()
        duree = (fin - debut).total_seconds()
        
        if result.returncode == 0:
            print(f"\n✅ {script['nom']} terminée avec succès ({duree:.1f}s)")
            resultats.append({'nom': script['nom'], 'statut': 'OK', 'duree': duree})
        else:
            print(f"\n⚠️ {script['nom']} terminée avec des avertissements")
            resultats.append({'nom': script['nom'], 'statut': 'WARNING', 'duree': duree})
            
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'exécution de {script['fichier']} :")
        print(f"   {str(e)}")
        resultats.append({'nom': script['nom'], 'statut': 'ERREUR', 'duree': 0})
        
        continuer = input("\n   Continuer malgré l'erreur ? (o/n) : ").strip().lower()
        if continuer != 'o':
            print("\n❌ Analyse interrompue par l'utilisateur")
            break

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print("\n" + "="*90)
print(" RÉSUMÉ DE L'EXÉCUTION ".center(90, "="))
print("="*90)

print("\n📊 STATUT DES ANALYSES :\n")
for i, res in enumerate(resultats, 1):
    if res['statut'] == 'OK':
        emoji = "✅"
    elif res['statut'] == 'WARNING':
        emoji = "⚠️"
    elif res['statut'] == 'SKIP':
        emoji = "⏭️"
    else:
        emoji = "❌"
    
    duree_str = f"{res['duree']:.1f}s" if res['duree'] > 0 else "N/A"
    print(f"   {emoji} {i}. {res['nom']:30s} : {res['statut']:10s} ({duree_str})")

# Vérification des fichiers générés
print(f"\n📁 GRAPHIQUES GÉNÉRÉS :")
graphiques_attendus = []
for script in scripts:
    graphiques_attendus.extend(script['outputs'])

graphiques_trouves = 0
graphiques_manquants = []

for graphique in graphiques_attendus:
    chemin = f"{OUTPUT_DIR}/{graphique}"
    if os.path.exists(chemin):
        taille = os.path.getsize(chemin) / 1024
        print(f"   ✓ {graphique:40s} ({taille:6.1f} KB)")
        graphiques_trouves += 1
    else:
        print(f"   ❌ {graphique:40s} (MANQUANT)")
        graphiques_manquants.append(graphique)

# Statistiques finales
duree_totale = sum(r['duree'] for r in resultats)
nb_ok = sum(1 for r in resultats if r['statut'] == 'OK')
nb_erreurs = sum(1 for r in resultats if r['statut'] == 'ERREUR')

print("\n" + "="*90)
print(" STATISTIQUES FINALES ".center(90, "="))
print("="*90)

print(f"\n✅ Analyses réussies : {nb_ok}/{len(scripts)}")
print(f"❌ Analyses échouées : {nb_erreurs}/{len(scripts)}")
print(f"📊 Graphiques générés : {graphiques_trouves}/{len(graphiques_attendus)}")
print(f"⏱️  Durée totale : {duree_totale:.1f} secondes ({duree_totale/60:.1f} minutes)")

if nb_ok == len(scripts) and graphiques_trouves == len(graphiques_attendus):
    print("\n" + "🎉" * 30)
    print("🎉 ANALYSE EXPLORATOIRE TERMINÉE AVEC SUCCÈS ! 🎉".center(90))
    print("🎉" * 30)
    print(f"\n📂 Tous les graphiques sont disponibles dans : {OUTPUT_DIR}/")
    print("\n🚀 PROCHAINES ÉTAPES :")
    print("   1. Analyser les visualisations générées")
    print("   2. Extraire les insights clés pour le pitch")
    print("   3. Préparer la modélisation (Jour 2)")
elif nb_ok > 0:
    print("\n⚠️ Analyse partiellement complétée")
    print(f"   → Vérifier les fichiers dans {OUTPUT_DIR}/")
else:
    print("\n❌ Aucune analyse n'a abouti")
    print("   → Vérifier les logs d'erreur ci-dessus")

print("\n" + "="*90)
print(f"🕐 Fin : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("="*90 + "\n")