# summit-analytics

Analyse statistique des expéditions de l'Himalaya népalais (1905–2025) : comment l'altitude, la saison, la taille de l'équipe et le sommet sont liés à la réussite et aux décès.

Le rapport complet est dans [`rapport.pdf`](rapport.pdf).

## Lancer l'analyse

```bash
pip install -r requirements.txt
python analyse.py
```

Les données viennent de la [Himalayan Database](https://himalayandatabase.com/). Exporte la table des expéditions en CSV et place-la à côté de `analyse.py` sous le nom `himalaya_expeditions_complet.csv`. Les graphiques sont enregistrés dans `figures/`.

*@mathematicien100frontieres, octobre 2026*
