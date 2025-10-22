#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCRIPT DE LANCEMENT COMPLET - HACKATHON
Exécute : Extraction coefficients → Lancement Streamlit
"""

import subprocess
import os
import sys
import socket

print("="*80)
print(" LANCEMENT PIPELINE COMPLET ".center(80, "="))
print("="*80)

# =============================================================================
# ÉTAPE 1 : EXTRACTION DES COEFFICIENTS
# =============================================================================

print("\n📊 [1/2] Extraction des coefficients de régression...")
print("-"*80)

try:
    result = subprocess.run(
        ["python", "scripts/simulation/coef.py"],
        check=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print("✅ Extraction terminée avec succès !")
except subprocess.CalledProcessError as e:
    print(f"❌ Erreur lors de l'extraction : {e}")
    print(e.stderr)
    sys.exit(1)

# =============================================================================
# DEBUG: VÉRIFIER LES PORTS OCCUPÉS
# =============================================================================

print("\n🔍 [DEBUG] Vérification des ports occupés...")
print("-"*80)

def check_port(port):
    """Vérifie si un port est libre"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True  # Port libre
    except OSError as e:
        return False  # Port occupé

# Tester les ports 8501-8510
for port in range(8501, 8511):
    status = "✅ LIBRE" if check_port(port) else "❌ OCCUPÉ"
    print(f"   Port {port}: {status}")

# =============================================================================
# FONCTION POUR TROUVER UN PORT LIBRE
# =============================================================================

def find_free_port(start_port=8501, max_attempts=10):
    """Trouve un port libre en commençant par start_port"""
    print(f"\n🔍 [DEBUG] Recherche d'un port libre de {start_port} à {start_port + max_attempts - 1}...")
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                print(f"   ✅ Port {port} est libre!")
                return port
        except OSError as e:
            print(f"   ❌ Port {port} occupé (erreur: {e})")
            continue
    
    raise RuntimeError(f"Impossible de trouver un port libre entre {start_port} et {start_port + max_attempts}")

# =============================================================================
# ÉTAPE 2 : LANCEMENT STREAMLIT
# =============================================================================

print("\n🚀 [2/2] Lancement de l'application Streamlit...")
print("-"*80)

try:
    port = find_free_port()
    print(f"\n🔌 [DEBUG] Port sélectionné : {port}")
    print(f"📱 L'application sera accessible sur : http://localhost:{port}")
    print("👉 Si elle ne s'ouvre pas automatiquement, copiez l'URL ci-dessus.\n")
    
    # Construire la commande
    cmd = [sys.executable, "-m", "streamlit", "run", "scripts/simulation/streamlit_app.py",
           "--server.port", str(port),
           "--server.headless", "true"]
    
    print(f"🔍 [DEBUG] Commande à exécuter:")
    print(f"   {' '.join(cmd)}\n")
    
    # Lancer Streamlit
    print("🚀 [DEBUG] Lancement en cours...\n")
    subprocess.run(cmd, check=True)
    
except KeyboardInterrupt:
    print("\n\n⏸️  Application arrêtée par l'utilisateur.")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Erreur lors du lancement Streamlit : {e}")
    print(f"🔍 [DEBUG] Code de sortie : {e.returncode}")
    sys.exit(1)
except RuntimeError as e:
    print(f"\n❌ {e}")
    print("\n💡 Solution : Fermez les autres instances avec:")
    print("   taskkill /F /IM python.exe")
    print("\nOu relancez manuellement:")
    print("   cd scripts/simulation")
    print("   streamlit run streamlit_app.py")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Erreur inattendue : {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)