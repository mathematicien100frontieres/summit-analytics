# -*- coding: utf-8 -*-
"""Summit Analytics : réussite et décès dans les expéditions de l'Himalaya
népalais, à partir de la Himalayan Database.

Lancer depuis la racine du projet : python analyse.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# SciPy sert seulement aux tests du khi-deux
try:
    from scipy.stats import chi2_contingency
except ImportError:
    chi2_contingency = None


# Chemins
DOSSIER = Path(__file__).resolve().parent
CSV_PATH = DOSSIER / "himalaya_expeditions_complet.csv"
OUTPUT_DIR = DOSSIER / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV introuvable : {CSV_PATH}\n"
        "Mets le CSV dans le même dossier que analyse.py (voir le README)."
    )

# Chargement
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)
print("Dimensions du fichier :", df.shape)
print("\nPremières lignes :")
print(df.head())
print("\nColonnes :")
print(df.columns.tolist())
print("\nValeurs manquantes (15 colonnes les plus touchées) :")
print(df.isna().sum().sort_values(ascending=False).head(15))

# Colonnes nécessaires
colonnes_requises = [
    "EXPID", "PEAKID", "YEAR", "SEASON_FACTOR", "HEIGHTM",
    "SUCCESS1", "SUCCESS2", "SUCCESS3", "SUCCESS4",
    "TOTMEMBERS", "TOTHIRED", "MDEATHS", "HDEATHS"
]
colonnes_absentes = [c for c in colonnes_requises if c not in df.columns]
if colonnes_absentes:
    raise KeyError(f"Colonnes absentes du CSV : {colonnes_absentes}")

# Réussite : au moins un des quatre SUCCESS est vrai
# (CLAIMED et DISPUTED ne sont pas pris en compte)
colonnes_succes = ["SUCCESS1", "SUCCESS2", "SUCCESS3", "SUCCESS4"]
df["REUSSITE_EXPEDITION"] = (
    df[colonnes_succes].fillna(False).astype(bool).any(axis=1)
)

# Décès : membres + personnes engagées
for colonne in ["HEIGHTM", "YEAR", "TOTMEMBERS", "TOTHIRED", "MDEATHS", "HDEATHS"]:
    df[colonne] = pd.to_numeric(df[colonne], errors="coerce")

df["DECES_TOTAL"] = (
    df["MDEATHS"].fillna(0) + df["HDEATHS"].fillna(0)
)
df["EXPEDITION_AVEC_DECES"] = df["DECES_TOTAL"] > 0


# Intervalle de Wilson à 95 %
def intervalle_wilson(nombre_succes, nombre_total, z=1.96):
    n = int(nombre_total)
    x = int(nombre_succes)
    if n <= 0:
        return np.nan, np.nan

    p = x / n
    denominateur = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominateur
    marge = (
        z
        * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        / denominateur
    )
    return max(0, centre - marge), min(1, centre + marge)


# Taux et intervalles pour chaque catégorie d'une colonne
def resume_par_groupe(colonne_groupe):
    lignes = []

    for categorie, groupe in df.groupby(
        colonne_groupe, observed=True, dropna=False
    ):
        n = int(groupe["EXPID"].nunique())
        nb_reussites = int(groupe["REUSSITE_EXPEDITION"].sum())
        nb_expeditions_avec_deces = int(
            groupe["EXPEDITION_AVEC_DECES"].sum()
        )
        nb_deces = int(groupe["DECES_TOTAL"].sum())

        ic_reussite = intervalle_wilson(nb_reussites, n)
        ic_deces = intervalle_wilson(nb_expeditions_avec_deces, n)

        lignes.append({
            "groupe": categorie,
            "expeditions": n,
            "reussites": nb_reussites,
            "taux_reussite_pct": 100 * nb_reussites / n if n else np.nan,
            "IC95_reussite_bas_pct": 100 * ic_reussite[0],
            "IC95_reussite_haut_pct": 100 * ic_reussite[1],
            "expeditions_avec_deces": nb_expeditions_avec_deces,
            "risque_deces_pct": (
                100 * nb_expeditions_avec_deces / n if n else np.nan
            ),
            "IC95_deces_bas_pct": 100 * ic_deces[0],
            "IC95_deces_haut_pct": 100 * ic_deces[1],
            "deces_enregistres": nb_deces,
        })

    if not lignes:
        return pd.DataFrame()
    return pd.DataFrame(lignes).set_index("groupe")


# Par saison
par_saison = resume_par_groupe("SEASON_FACTOR")
print("\nRésultats par saison :")
print(par_saison.round(1).to_string())

# Par sommet (les 15 plus fréquentés)
par_sommet = resume_par_groupe("PEAKID")
print("\n15 sommets les plus fréquentés :")
print(
    par_sommet.sort_values("expeditions", ascending=False)
    .head(15)
    .round(1)
    .to_string()
)

# Par tranche d'altitude
bornes_altitude = [-np.inf, 6500, 7000, 7500, 8000, 8500, np.inf]
noms_altitude = [
    "< 6500 m", "6500–6999 m", "7000–7499 m",
    "7500–7999 m", "8000–8499 m", "8500 m et plus"
]
df["TRANCHE_ALTITUDE"] = pd.cut(
    df["HEIGHTM"],
    bins=bornes_altitude,
    labels=noms_altitude,
    right=False
)

par_altitude = resume_par_groupe("TRANCHE_ALTITUDE")
print("\nRésultats par altitude :")
print(par_altitude.round(1).to_string())

# Par taille d'équipe (membres + personnes engagées)
df["TAILLE_EQUIPE_ALTITUDE"] = df["TOTMEMBERS"] + df["TOTHIRED"]

df["TRANCHE_EQUIPE"] = pd.cut(
    df["TAILLE_EQUIPE_ALTITUDE"],
    bins=[-np.inf, 5, 10, 20, 50, np.inf],
    labels=["≤ 5", "6–10", "11–20", "21–50", "> 50"],
    include_lowest=True
)

par_equipe = resume_par_groupe("TRANCHE_EQUIPE")
print("\nRésultats par taille d'équipe :")
print(par_equipe.round(1).to_string())


# Graphique : taux de réussite + intervalle de confiance
def tracer_reussite(tableau, titre, etiquette_x, nom_fichier):
    tableau = tableau.dropna(
        subset=[
            "taux_reussite_pct",
            "IC95_reussite_bas_pct",
            "IC95_reussite_haut_pct"
        ]
    )
    if tableau.empty:
        print(f"Pas de données pour : {titre}")
        return

    x = np.arange(len(tableau))
    taux = tableau["taux_reussite_pct"].to_numpy() / 100
    bas = tableau["IC95_reussite_bas_pct"].to_numpy() / 100
    haut = tableau["IC95_reussite_haut_pct"].to_numpy() / 100

    plt.figure(figsize=(9, 5))
    plt.errorbar(
        x, taux,
        yerr=np.vstack([taux - bas, haut - taux]),
        fmt="o-",
        capsize=5
    )
    plt.xticks(x, tableau.index.astype(str), rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylabel("Proportion d'expéditions réussies")
    plt.xlabel(etiquette_x)
    plt.title(titre)
    plt.tight_layout()

    chemin_image = OUTPUT_DIR / nom_fichier
    plt.savefig(chemin_image, dpi=200, bbox_inches="tight")
    print(f"Graphique enregistré : {chemin_image}")
    plt.show()
    plt.close()


tracer_reussite(
    par_altitude,
    "Réussite selon l'altitude du sommet (IC à 95 %)",
    "Altitude officielle du sommet",
    "reussite_par_altitude.png"
)

# La saison inconnue (2 expéditions) n'est pas tracée
par_saison_graphique = par_saison.drop(index="Unknown", errors="ignore")
tracer_reussite(
    par_saison_graphique,
    "Réussite selon la saison (IC à 95 %)",
    "Saison",
    "reussite_par_saison.png"
)

tracer_reussite(
    par_equipe,
    "Réussite selon la taille de l'équipe (IC à 95 %)",
    "Membres et personnes engagées au-dessus du camp de base",
    "reussite_par_taille_equipe.png"
)


# Test du khi-deux entre une catégorie et un résultat
# (association brute : ça ne prouve pas une cause)
def tester_association(colonne_groupe, colonne_resultat, description, filtre):
    donnees = df.loc[filtre, [colonne_groupe, colonne_resultat]].dropna()
    tableau = pd.crosstab(
        donnees[colonne_groupe], donnees[colonne_resultat]
    )

    if tableau.shape[0] < 2 or tableau.shape[1] < 2:
        print(f"\nTest impossible : {description}")
        return

    chi2, p_value, degres_liberte, effectifs_attendus = chi2_contingency(
        tableau
    )
    print(f"\nKhi-deux, {description}")
    print(f"Khi-deux = {chi2:.2f} ; ddl = {degres_liberte}")
    print(f"Valeur p = {p_value:.4g}")

    if (effectifs_attendus < 5).any():
        print("Attention : un effectif attendu est inférieur à 5.")


if chi2_contingency is None:
    print("\nSciPy absent : pas de tests du khi-deux.")
else:
    saison_connue = df["SEASON_FACTOR"] != "Unknown"
    altitude_connue = df["TRANCHE_ALTITUDE"].notna()
    equipe_connue = df["TRANCHE_EQUIPE"].notna()

    tester_association(
        "SEASON_FACTOR", "REUSSITE_EXPEDITION",
        "saison et réussite", saison_connue
    )
    tester_association(
        "SEASON_FACTOR", "EXPEDITION_AVEC_DECES",
        "saison et expédition avec décès", saison_connue
    )
    tester_association(
        "TRANCHE_ALTITUDE", "REUSSITE_EXPEDITION",
        "altitude et réussite", altitude_connue
    )
    tester_association(
        "TRANCHE_ALTITUDE", "EXPEDITION_AVEC_DECES",
        "altitude et expédition avec décès", altitude_connue
    )
    tester_association(
        "TRANCHE_EQUIPE", "REUSSITE_EXPEDITION",
        "taille de l'équipe et réussite", equipe_connue
    )
    tester_association(
        "TRANCHE_EQUIPE", "EXPEDITION_AVEC_DECES",
        "taille de l'équipe et expédition avec décès", equipe_connue
    )


# Aperçu des notes d'accidents et de fin d'expédition
for colonne in ["ACCIDENTS", "TERMNOTE"]:
    print(f"\nExemples de la colonne {colonne} :")
    print("Lignes renseignées :", df[colonne].notna().sum())

    exemples = df[["YEAR", "PEAKID", colonne]].dropna(subset=[colonne]).head(10)
    print(exemples.to_string(index=False))

print("\nAnalyse terminée. Les graphiques sont dans figures/.")
    
    