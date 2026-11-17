# Noria — Fiche projet

> **Évaluer et expliquer le risque de crédit des profils sans historique bancaire.**
> Projet web + IA · Licence informatique (SALEAM-IA), NIT · Racha CHABANE · Rendu fin décembre 2026
> Dernière mise à jour : 17 novembre 2026 (après l'EDA, session 1)

---

## 1. Problème et solution

**Problème.** En Algérie, les jeunes actifs, freelances et auto-entrepreneurs se voient souvent refuser un crédit faute d'historique bancaire. Le refus est une boîte noire : ils ne savent ni pourquoi, ni quoi améliorer. Côté banque, le conseiller manque d'outils objectifs pour évaluer ces profils atypiques.

**Solution.** Une application web qui s'appuie sur des modèles de machine learning entraînés :
- un modèle **client**, pour une estimation publique et anonyme ;
- un modèle **banque**, plus précis, qui calcule un score de risque expliqué pour le conseiller.

L'outil recommande, l'humain décide : **aucune décision n'est automatique.**

---

## 2. Utilisateurs

| Rôle | Persona | Besoin | Sur Noria |
|---|---|---|---|
| **Client** (public) | Amine, 26 ans, développeur freelance à Blida, revenus irréguliers, aucun crédit passé | « Je veux savoir si j'ai mes chances avant de me déplacer. » | Sans compte, sur téléphone : formulaire en 3 étapes → catégorie (favorable / à renforcer / fragile) + pistes concrètes. Rien n'est stocké ni envoyé à la banque. |
| **Conseiller** | Samia, 38 ans, conseillère depuis 8 ans, ~20 dossiers par semaine | « Si l'outil dit non, je dois pouvoir l'expliquer au client. » | Connectée : saisit les informations vérifiées → probabilité de défaut, classe de risque, facteurs explicatifs, recommandation → enregistre sa décision. |
| **Administrateur** | Directeur d'agence ou responsable IT | Contrôler l'outil | Crée les comptes conseillers, voit tous les dossiers, consulte le journal d'audit. |

**11 pages :** 3 côté client, 5 côté conseiller (connexion partagée avec l'admin), 4 côté admin. Les deux côtés sont indépendants : le client ne soumet rien à la banque depuis l'application.

---

## 3. Données

**Source :** Home Credit Default Risk (Home Credit Group, compétition Kaggle 2018). Table utilisée : `application_train.csv`.

**Chiffres clés (EDA, session 1) :**
- 307 511 demandes × 122 colonnes (106 numériques, 16 catégorielles)
- `TARGET` : **8,07 % de défauts** (ratio 1:11,4) → classes déséquilibrées
- 67 colonnes avec des valeurs manquantes, dont 41 à plus de 50 % (surtout les colonnes logement)
- Les `EXT_SOURCE_1/2/3` sont de loin les variables les plus corrélées à la cible

**Pertinence :** Home Credit prête justement à des populations avec peu ou pas d'historique de crédit.

### Choix sur les données
- **Seule la table principale sert à l'entraînement.** Les tables d'historique (bureau de crédit, paiements passés) sont exclues des features, pour rester cohérent avec la cible « sans historique ».
- **`bureau.csv` sert uniquement à l'évaluation :** les clients du jeu de test absents de `bureau.csv` forment le sous-groupe « sans historique » sur lequel la performance est mesurée séparément. Il ne sert jamais à entraîner.
- **Seuls les prêts classiques (`Cash loans`) sont conservés.** Noria simule un crédit défini par un montant et une durée ; pour les crédits renouvelables (~10 %), la durée n'a pas de sens.

---

## 4. Préparation des données

| Étape | Contenu |
|---|---|
| **Filtrage** | Cash loans uniquement ; suppression des lignes `CODE_GENDER = XNA` (4) et `NAME_FAMILY_STATUS = Unknown` (2) |
| **Nettoyage** | `DAYS_EMPLOYED = 365243` (55 374 lignes, retraités et chômeurs) → indicateur `EST_RETRAITE`, puis valeur mise en manquant ; suppression des colonnes logement normalisées (~47 colonnes) ; regroupement des modalités rares dans « Autre » |
| **Variables créées** | Âge (ans), ancienneté (ans), crédit / revenu, mensualité / revenu (taux d'effort), durée ≈ crédit ÷ mensualité, revenu par personne, ancienneté / âge |
| **Split figé** | Train 70 % · validation 15 % · test 15 %, stratifié sur `TARGET`, `random_state = 42`, `SK_ID_CURR` conservé |
| **Pipeline (fit sur le train uniquement)** | Imputation (médiane / modalité la plus fréquente), plafonnement des extrêmes (1er–99e percentile), encodage one-hot (catégories inconnues ignorées), normalisation pour la régression logistique |

**Pourquoi des ratios ?** Ils rendent le modèle moins dépendant de la devise et du niveau de vie du pays d'origine.

**Règle anti-fuite :** tout ce qui apprend des données (médianes, percentiles, encodage) est appris sur le train seulement, dans un pipeline scikit-learn reproductible.

---

## 5. Les modèles

Trois jeux de variables, définis dans `src/config.py` :

| | **Client (public)** | **Banque sans historique** | **Banque avec historique** |
|---|---|---|---|
| **Rôle** | Estimation rapide sur des données déclarées, non vérifiées | Score précis sur des données vérifiées, sans historique | Référence : mesurer ce qu'apporte l'historique |
| **Variables** | Âge, situation familiale, enfants, type d'emploi, ancienneté, retraité, crédit / revenu, durée, mensualité / revenu | Toutes les variables pertinentes, **sauf** `EXT_SOURCE_*`, `AMT_REQ_CREDIT_BUREAU_*` et leurs indicateurs de manquant | Banque sans historique + `EXT_SOURCE_*` + `AMT_REQ_CREDIT_BUREAU_*` |
| **Sortie** | Catégorie favorable / à renforcer / fragile, points forts et points à renforcer | Probabilité de défaut calibrée, classe de risque faible / moyen / élevé, recommandation accepter / sous conditions / refuser | Utilisé pour la comparaison |

**Algorithmes :** LightGBM (modèle principal), comparé à une régression logistique de référence qui doit être battue. Bonus : XGBoost et CatBoost pour un tableau comparatif.

**Entraînement :** pondération des classes, validation croisée stratifiée à 5 plis, réglage des hyperparamètres, early stopping.

**Calibration :** isotonic ou Platt, pour qu'une probabilité affichée de 20 % corresponde à environ 20 % de défauts réels.

**Explicabilité :** SHAP. Les 5 à 10 facteurs qui ont pesé sur chaque dossier, traduits en français clair pour le conseiller, et en phrases simples pour le client.

**Contrainte du modèle client :** chaque variable doit être calculable à partir des seules saisies du formulaire (la mensualité vaut montant ÷ durée). Le dataset donne un revenu **annuel**, le formulaire demande un revenu **mensuel** : conversion ×12 dans l'API.

---

## 6. Évaluation

- **Métrique principale : AUC-ROC** (adaptée au déséquilibre, contrairement à l'accuracy), complétée par l'AUC-PR, le rappel sur les défauts et la matrice de confusion.
- **Courbe de calibration.**
- **Jeu de test utilisé une seule fois**, à la fin.
- **Écart banque avec / sans historique / client mesuré et affiché : c'est la contribution originale du projet.** Il montre ce que coûtent, en précision, l'absence d'historique et une information non vérifiée.
- **Performance sur le sous-groupe « sans historique »** (clients du test absents de `bureau.csv`), comparée à la performance globale.
- **Performance par type d'emploi et par âge** (`Working`, `Commercial associate` comme profil le plus proche des freelances, 20-35 ans), pour vérifier que le modèle reste utilisable pour la population cible.
- **Règle de décision pour le modèle client, fixée avant de voir les résultats** (AUC sur la validation) : ≥ 0,65 → conservé ; entre 0,60 et 0,65 → conservé avec un message de prudence renforcé ; < 0,60 → abandonné, et la décision est documentée.

---

## 7. Rigueur et éthique

- **Genre exclu** de toutes les variables du modèle, conservé uniquement pour l'audit d'équité (écart brut observé à l'EDA : F 7,0 % contre M 10,1 % de défauts).
- **Audit d'équité** (Fairlearn) par sexe et tranche d'âge : taux d'acceptation et taux d'erreur comparés.
- **Historique caché exclu** du modèle sans historique : `EXT_SOURCE_*`, `AMT_REQ_CREDIT_BUREAU_*`, et le fait même qu'une de ces valeurs manque (lui aussi lié au défaut).
- Contrôle des fuites de données, pipeline reproductible, modèles versionnés.

---

## 8. Application et sécurité

**Architecture :** front React + Tailwind · API FastAPI (Python) · base PostgreSQL. Les modèles sont chargés en mémoire au démarrage de l'API et servis en temps réel (réponse en moins de 2 s).

**Sécurité :** authentification JWT, mots de passe hashés avec bcrypt, contrôle d'accès par rôle côté serveur, validation stricte des entrées (Pydantic), ORM contre les injections, journal d'audit, limitation des tentatives de connexion.

**Côté public :** minimisation des données (aucune simulation conservée), limitation du nombre de requêtes par IP, pas d'inscription publique (seul l'admin crée les comptes).

---

## 9. Formulaire client (aligné sur le modèle)

| Étape | Champs |
|---|---|
| **1 · Profil** | Âge, situation familiale, nombre d'enfants à charge |
| **2 · Activité** | Type d'emploi (salarié, freelance, auto-entrepreneur, sans emploi, retraité), ancienneté, revenu mensuel (DA) |
| **3 · Projet** | Montant demandé (DA), durée (mois) |

Retirés par rapport à la version initiale : **apport personnel** et **objet du crédit** (absents du dataset, donc inutilisables par le modèle).

Correspondance avec le dataset : « freelance » et « auto-entrepreneur » sont associés à la catégorie la plus proche (`Commercial associate`). C'est une approximation, documentée dans les limites.

---

## 10. Périmètre et planning

**MVP :** les modèles et SHAP, simulation publique, espace conseiller complet (dossiers, score, décision, tableau de bord), espace admin, sécurité.

**Bonus, dans l'ordre :** audit d'équité, comparaison XGBoost / CatBoost, fourchette client, pistes d'amélioration (plus petit changement de montant ou de durée qui fait passer à la catégorie supérieure), page de transparence du modèle, chiffrement en base, résultat en PDF.

| Semaines | Travail |
|---|---|
| 1-4 | Données et modèles, en parallèle du design des maquettes |
| 5-6 | Backend et sécurité |
| 7-8 | Frontend · **MVP complet le 6 décembre** |
| 9-10 | Bonus, tests, rapport, une semaine tampon |

---

## 11. Limites assumées

- **Les données ne sont pas algériennes.** Noria est une preuve de concept : la méthode (ratios, double modèle, explicabilité, équité) est transposable, mais un usage réel nécessiterait un réentraînement sur des données locales.
- **La population cible est sous-représentée.** Le dataset n'a aucune catégorie freelance ou auto-entrepreneur (seulement 10 « Businessman »), et 51,6 % des demandeurs sont salariés. Les prédictions pour des profils comme Amine sont donc moins fiables.
- **Le modèle sans historique est moins précis**, puisqu'il perd les variables les plus prédictives. Cet écart est mesuré et constitue un résultat du projet, pas un défaut caché.

---

## 12. Journal des décisions

| Date | Décision | Raison |
|---|---|---|
| 17/11/2026 | Variable **apport** abandonnée | Prix du bien inférieur ou égal au crédit dans 99,9 % des lignes : « prix − crédit » est négatif ou nul presque partout |
| 17/11/2026 | **Cash loans** uniquement | Pour les crédits renouvelables, crédit ÷ mensualité vaut exactement 20 en médiane : ce n'est pas une vraie durée |
| 17/11/2026 | `AMT_REQ_CREDIT_BUREAU_*` traités comme de l'historique | Ce sont des demandes au bureau de crédit |
| 17/11/2026 | Indicateurs de manquant des `EXT_SOURCE` exclus du modèle sans historique | L'absence d'une `EXT_SOURCE` est elle-même liée au défaut (9,31 % contre 7,77 % pour `EXT_SOURCE_3`) |
| 17/11/2026 | Indicateur **`EST_RETRAITE`** créé avant d'effacer `365243` | Ces lignes ont un taux de défaut nettement plus bas (5,40 % contre 8,66 %) : l'information compte |
| 17/11/2026 | Colonnes logement normalisées supprimées | Vides sur ~47 % des lignes et inconnues du client comme du conseiller |
| 17/11/2026 | `bureau.csv` utilisé **pour l'évaluation seulement** | Prouver la performance sur les vrais profils sans historique, sans contredire le choix « table principale uniquement » |
| 17/11/2026 | **Objet du crédit** retiré du formulaire | Absent de la table principale |
| 17/11/2026 | Modèle client **conservé jusqu'à mesure** | Son abandon serait décidé sur des chiffres (règle d'AUC, section 6), pas sur une intuition |
| 17/11/2026 | Pas de ré-entraînement sur un dataset de freelances | Aucun dataset public fiable identifié (beaucoup sont synthétiques) ; on mesure plutôt la performance par type d'emploi |
| 17/11/2026 | Suite en PFE abandonnée | Données locales (ANADE, ANGEM, microfinance) très difficiles à obtenir ; reformulée en perspective |