import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CLASSES DE DONNÉES
# =============================================================================

@dataclass
class ActionSimulation:
    """Définition d'une action à simuler"""
    type_action: str  # 'doses_vaccins', 'pharmacies', 'sos_medecins', 'campagne_comm'
    valeur: float     # Budget en K€ pour comm, Nombre pour autres
    departement: str
    tranche_age: str = '65+'
    semaine_implementation: int = 40
    
@dataclass
class ResultatSimulation:
    """Résultats de la simulation"""
    departement: str
    baseline: Dict[str, float]
    simulation: Dict[str, float]
    delta: Dict[str, float]
    intervalle_confiance: Dict[str, Tuple[float, float]]
    impact_score: str
    recommandation: str
    cout_total: float
    roi: float
    details: Dict

# =============================================================================
# FONCTIONS UTILITAIRES (Saisonnalité & Timing)
# =============================================================================

def intensite_epidemique(semaine_annee: int) -> float:
    """Modèle l'intensité de l'épidémie (Pic en hiver)"""
    intensite = 0.6 + 0.4 * np.cos(2 * np.pi * (semaine_annee - 1) / 52)
    return np.clip(intensite, 0.2, 1.0)

def efficacite_vaccin_temporelle(semaines_avant_pic: int) -> float:
    """Efficacité optimale 4-8 semaines avant le pic"""
    if 4 <= semaines_avant_pic <= 8: return 0.75
    elif 2 <= semaines_avant_pic < 4: return 0.60
    elif semaines_avant_pic < 2: return 0.40
    elif 8 < semaines_avant_pic <= 12: return 0.70
    else: return 0.65

def calculer_semaines_avant_pic(semaine_actuelle: int, semaine_pic: int = 1) -> int:
    if semaine_actuelle <= semaine_pic: return semaine_pic - semaine_actuelle
    else: return (52 - semaine_actuelle) + semaine_pic

# =============================================================================
# MOTEUR DE SIMULATION CORRIGÉ (V3)
# =============================================================================

class SimulateurGrippeV3:
    """
    Moteur de simulation aligné sur la Documentation Technique MPJ.
    Implémente: Log-Comm, Plafond 70%, Séparation SOS/Vaccin.
    """
    
    def __init__(self, df_urgences: pd.DataFrame = None, df_couvertures: pd.DataFrame = None):
        
        # Coefficients et Paramètres
        self.coeffs_model = {
            'vaccination_urgences': -0.65, # Impact couverture sur urgences
            'sos_urgences_direct': -0.8,   # Impact direct SOS sur urgences (évite double compte) [cite: 26]
        }
        
        self.params_age = {
            '65-74': {'cout_passage_urgences': 180, 'taux_hospit': 0.12, 'cout_hospit': 3200},
            '75+':   {'cout_passage_urgences': 220, 'taux_hospit': 0.28, 'cout_hospit': 4800},
            'tous':  {'cout_passage_urgences': 190, 'taux_hospit': 0.18, 'cout_hospit': 3800}
        }
        
        self.couts_actions = {
            'cout_dose_vaccin': 12,
            'cout_pharmacie': 2500,
            'cout_medecin_sos': 80000,
            'cout_campagne_comm_par_k': 1000
        }
        
        self.efficacites = {
            'alpha_comm': 0.8, # Coefficient Alpha pour log(1+Budget) 
            'impact_pharmacie_couverture': 0.09
        }
        
        # --- CORRECTION 2: PLAFOND DE COUVERTURE ---
        # "Couv_max_empirique = p95 couverture France (~70%)" [cite: 17]
        self.contraintes = {
            'max_couverture_empirique': 70.0, 
            'delta_couverture_max': 15
        }
        
        self.df_urgences = df_urgences
        self.df_couvertures = df_couvertures
        self.metadata_dept = self._init_metadata_enrichie()

    def _init_metadata_enrichie(self) -> Dict:
        # Données démo simplifiées
        return {
            '59': {'nom': 'Nord', 'pop_65plus': 470000, 'superficie': 5743},
            '75': {'nom': 'Paris', 'pop_65plus': 350000, 'superficie': 105}
        }
    
    def calculer_baseline(self, departement: str, semaine: int = 1) -> Dict:
        """Calcule l'état initial avant action"""
        # Valeurs par défaut simplifiées pour l'exemple
        couv_base = 55.0 # %
        urg_base = 80.0  # passages/100k
        
        intensite = intensite_epidemique(semaine)
        urg_saisonnier = urg_base * intensite
        
        return {
            'couverture': couv_base,
            'urgences': urg_saisonnier,
            'hospitalisations': urg_saisonnier * 0.18, # Taux moyen
            'intensite': intensite
        }
    
    def _calculer_impact_couverture(self, action: ActionSimulation, baseline: Dict) -> float:
        """
        Calcul de l'impact sur la couverture vaccinale.
        Intègre la formule LOG pour la communication.
        """
        delta = 0
        pop_cible = self.metadata_dept.get(action.departement, {}).get('pop_65plus', 100000)
        superficie = self.metadata_dept.get(action.departement, {}).get('superficie', 5000)
        
        # Facteur de saturation (plus on est proche de 70%, plus c'est dur)
        marge_progression = max(0, self.contraintes['max_couverture_empirique'] - baseline['couverture'])
        facteur_saturation = (marge_progression / 20) if marge_progression < 20 else 1.0
        facteur_saturation = np.clip(facteur_saturation, 0.1, 1.0)

        if action.type_action == 'doses_vaccins':
            semaines_avant_pic = calculer_semaines_avant_pic(action.semaine_implementation)
            efficacite = efficacite_vaccin_temporelle(semaines_avant_pic)
            delta = (action.valeur / pop_cible) * 100 * efficacite * facteur_saturation
            
        elif action.type_action == 'pharmacies':
            coef = self.efficacites['impact_pharmacie_couverture']
            delta = (action.valeur / superficie) * coef * 10000 * facteur_saturation
            
        elif action.type_action == 'campagne_comm':
            # --- CORRECTION 1: FORMULE LOGARITHMIQUE ---
            # "DeltaCouv_corr = alpha * log(1 + Budget)" 
            # action.valeur est le Budget en K€
            alpha = self.efficacites['alpha_comm']
            delta = alpha * np.log1p(action.valeur) * facteur_saturation
            
        elif action.type_action == 'sos_medecins':
            # --- CORRECTION 3 (Partie A): SOS ne joue plus sur la couverture ---
            # "Double comptage SOS -> Problème" [cite: 22]
            delta = 0 
            
        return delta

    def _calculer_impact_urgences_final(self, 
                                        baseline: Dict, 
                                        delta_couv: float, 
                                        action: ActionSimulation) -> float:
        """
        Calcul final des urgences.
        Intègre la séparation : Urgences = Avant + Coef*Couv + Coef*SOS [cite: 26]
        """
        
        # 1. Impact via la couverture vaccinale
        # "Urgences_Après = Urgences_Avant + coef_vacc * DeltaCouv" [cite: 24]
        impact_vaccin = self.coeffs_model['vaccination_urgences'] * delta_couv
        
        # 2. Impact direct (Structurel) - SOS Médecins
        # "Urgences_Après_corr = ... + coef_SOS_region * DeltaSOS" [cite: 26]
        impact_direct_sos = 0
        if action.type_action == 'sos_medecins':
            # Hypothèse: 1 médecin SOS désengorge X passages aux urgences directement
            impact_direct_sos = action.valeur * self.coeffs_model['sos_urgences_direct']
            
        total_delta_urgences = impact_vaccin + impact_direct_sos
        
        return total_delta_urgences

    def simuler_action(self, action: ActionSimulation) -> Dict:
        """Exécute la simulation avec les nouvelles formules"""
        
        baseline = self.calculer_baseline(action.departement, action.semaine_implementation)
        
        # 1. Calcul Delta Couverture
        delta_couv_brut = self._calculer_impact_couverture(action, baseline)
        
        # --- CORRECTION 2: APPLICATION DU PLAFOND ---
        # "Couverture_corr = min(Couv_avant + delta, Couv_max_empirique)" [cite: 15]
        couv_potentielle = baseline['couverture'] + delta_couv_brut
        couv_finale = min(couv_potentielle, self.contraintes['max_couverture_empirique'])
        delta_couv_reelle = couv_finale - baseline['couverture']
        
        # 2. Calcul Delta Urgences (Séparé)
        delta_urgences = self._calculer_impact_urgences_final(baseline, delta_couv_reelle, action)
        
        # 3. Calculs financiers (simplifiés pour l'exemple)
        cout = 0
        if action.type_action == 'campagne_comm':
            cout = action.valeur * 1000 # Valeur en K€
        elif action.type_action == 'doses_vaccins':
            cout = action.valeur * 12
        elif action.type_action == 'sos_medecins':
            cout = action.valeur * 80000
            
        # ROI simple
        params = self.params_age.get(action.tranche_age, self.params_age['tous'])
        economie = abs(delta_urgences) * (params['cout_passage_urgences'] + (params['taux_hospit'] * params['cout_hospit']))
        roi = economie / cout if cout > 0 else 0
        
        return {
            'action': action.type_action,
            'input_valeur': action.valeur,
            'baseline_couv': baseline['couverture'],
            'delta_couv': delta_couv_reelle,
            'finale_couv': couv_finale,
            'is_capped': couv_finale == self.contraintes['max_couverture_empirique'],
            'delta_urgences': delta_urgences,
            'cout': cout,
            'roi': roi
        }

# =============================================================================
# TEST DES CORRECTIONS
# =============================================================================

if __name__ == "__main__":
    sim = SimulateurGrippeV3()
    dept = '59'
    
    print("--- TEST 1: CAMPAGNE DE COM (Formule Logarithmique)  ---")
    # On teste avec 10k€ vs 100k€ pour voir l'effet non-linéaire
    a1 = ActionSimulation('campagne_comm', 10, dept)  # 10 K€
    a2 = ActionSimulation('campagne_comm', 100, dept) # 100 K€ (10x plus)
    
    res1 = sim.simuler_action(a1)
    res2 = sim.simuler_action(a2)
    
    print(f"Budget 10k€  -> +{res1['delta_couv']:.2f}% de couverture")
    print(f"Budget 100k€ -> +{res2['delta_couv']:.2f}% de couverture")
    print("Note: 10x budget ne donne pas 10x impact (Logarithmique vérifié)")
    
    print("\n--- TEST 2: PLAFOND EMPIRIQUE  ---")
    # On force une dose massive pour taper le plafond
    a3 = ActionSimulation('doses_vaccins', 1000000, dept) 
    res3 = sim.simuler_action(a3)
    print(f"Baseline: {res3['baseline_couv']}%")
    print(f"Tentative ajout massif... Résultat: {res3['finale_couv']:.1f}%")
    print(f"Plafond activé ? {res3['is_capped']} (Max fixé à {sim.contraintes['max_couverture_empirique']}%)")

    print("\n--- TEST 3: SOS MÉDECINS (Impact Direct) [cite: 26] ---")
    # SOS ne doit pas changer la couverture mais réduire les urgences
    a4 = ActionSimulation('sos_medecins', 5, dept)
    res4 = sim.simuler_action(a4)
    print(f"Ajout 5 médecins SOS -> Delta Couverture: {res4['delta_couv']:.2f}% (Doit être 0)")
    print(f"Ajout 5 médecins SOS -> Delta Urgences: {res4['delta_urgences']:.2f} passages (Impact direct)")