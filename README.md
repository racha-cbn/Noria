# Noria

Outil web de scoring crédit expliqué pour les profils avec peu ou pas d'historique bancaire (projet web + IA de licence, dataset Kaggle *Home Credit Default Risk*).

**Fiche de référence du projet : [docs/fiche-projet.md](docs/fiche-projet.md)** (choix de données, modèles, éthique, split, et journal des décisions en section 12).

## Structure

```
noria/
├── docs/
│   └── fiche-projet.md   # Fiche projet de référence + journal des décisions
├── data/
│   ├── raw/              # Données brutes (jamais modifiées, non versionnées)
│   └── processed/        # Données préparées (non versionnées)
├── notebooks/            # Exploration et expérimentations (01_eda_application.ipynb…)
├── src/                  # Code Python réutilisable (préparation, modèles)
├── reports/
│   └── figures/          # Graphiques générés par les notebooks
├── requirements.txt
└── README.md
```
