**BRIEF PROJET : Prédiction du risque de santé mentale chez les étudiants**

**1\. Contexte du projet**

Un établissement d’enseignement souhaite analyser un jeu de données étudiant afin d’étudier les facteurs associés au bien-être et au risque de santé mentale, puis construire un prototype de classification supervisée. Le projet couvre toute la chaîne Data Science : audit, nettoyage, EDA, feature engineering, preprocessing, entraînement, optimisation, évaluation, automatisation et visualisation.

Le projet est strictement pédagogique. Le modèle ne doit jamais être présenté comme un outil de diagnostic, de dépistage clinique, de décision disciplinaire ou de sélection d’étudiants.

**DataSet:** https\://www\.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students

**2\. Objectifs métier** 

* Détection des risques : Identifier les étudiants présentant un risque accru de difficultés psychologiques.  
* Prévention : Anticiper les besoins d’accompagnement.  
* Analyse des facteurs : Comprendre les facteurs associés aux risques psychologiques.  
* Accompagnement : Orienter les étudiants vers des ressources adaptées.  
* Optimisation : Prioriser les actions de soutien étudiant.

**3\. Étapes à réaliser** 

### **Étape 1 — Chargement et Analyse exploratoire des données** 

* comprendre la structure générale du jeu de données (types de données, dimensions, aperçus) ;  
* Vérifier les valeurs manquantes, les doublons, les incohérences et les valeurs aberrantes.   
* Produire des statistiques descriptives.  
* Étudier la distribution de la variable cible et l’éventuel déséquilibre des classes.  
* Analyser les relations entre les variables explicatives et la cible.  
* Examiner les corrélations entre les variables.  
* identifier les variables les plus fortement liées au niveau de risque ;

#### **Étape 2 — Transformation et préparation des données**

* Diviser les données en deux ensembles : un ensemble d’entraînement (80 %) et un ensemble de test (20 %).  
* Traiter les valeurs manquantes en choisissant la méthode d’imputation la plus adaptée à la nature des données  
* Détecter et traiter les valeurs aberrantes en utilisant une méthode appropriée (IQR, Z-score, etc.).  
* Encoder les variables catégorielles afin de les convertir en valeurs numériques  
* Gérer le déséquilibre des classes avec des techniques de sur-échantillonnage ou sous-échantillonnage (Smote), si nécessaire.   
* Normaliser ou standardiser les variables numériques afin de les mettre à la même échelle.  
* Regrouper les transformations dans un pipeline afin d’éviter le data leakage. 

**Chaque décision de nettoyage doit être justifiée.**

### **Étape 3 — Entraînement et comparaison des modèles**

* Entraîner au moins 3 modèles de classification.  
* Utiliser les mêmes données d’entraînement pour les comparer.  
* Évaluer les modèles avec des métriques adaptées :accuracy, précision, rappel, F1-score, matrice de confusion et, si pertinent, ROC-AUC.

##### **Étape 4 — Optimisation et évaluation**

* Sélectionner les modèles les plus prometteurs.  
* Utiliser des techniques comme GridSearchCV ou RandomizedSearchCV pour affiner les hyperparamètres des modèles et améliorer leurs performances.  
* Comparer les performances des différents modèles.  
* Choisir le modèle le plus adapté au contexte métier.

* Évaluer les performances du modèle optimisé sur le jeu de test indépendant à l'aide de métriques adaptées et d'une matrice de confusion.

* Analyser les erreurs de prédiction et les limites du modèle.

* Présenter les résultats et justifier le choix du modèle.

* Sauvegarder le pipeline complet (prétraitement \+ modèle) au format `.pkl` ou `.joblib`. 

##### **Étape 5 — Orchestration avec Airflow & application Streamlit**

* Concevoir un DAG Airflow permettant d'orchestrer et d'automatiser les différentes étapes du pipeline : préparation des données, prétraitement, entraînement et évaluation du modèle.  
* Prévoir l'exécution planifiée du DAG ainsi que le suivi de son état et de ses éventuelles erreurs.  
* Développer une application interactive avec Streamlit permettant de consulter les performances du modèle (métriques, graphiques et matrice de confusion) et de tester le modèle sur de nouvelles données.

## **4\. Modalités pédagogiques**

* Brief : Jury Blanc  
* Travail individuel  
* Temporalité : 5 jours  
* Période : Du 09/11/2026 au 14/11/2026 à 23h59


## **5\. Modalités d'évaluation**

* Date de soutenance (16/11/2026)  
* Durée de soutenance jury blanc (45 min), organisée comme suit :   
  * Présentation rapide (10 minutes) : Démonstration du contenu et des fonctionnalités principales de l’application.  
  * Analyse du code (10 minutes) : Montrez et expliquez le code source, en soulignant vos choix techniques.  
  * Mise en situation (20 minutes) : L’examinateur propose des cas d’usage pour tester votre application.  
  * Code Review et questions techniques (5 minutes).

## **6\. Livrables**

* Un dépôt Git organisé et documenté, contenant :  
  * le notebook complet / scripts Python  ;  
  * le code source structuré (data/, /dag, notebooks/, src/, models/, app/) ;  
  * le fichier requirements.txt ;  
  * le modèle entraîné et le scaler sérialisés (.pkl / .joblib) ;  
  * l'application Streamlit ;  
  * un README clair : description du projet, dataset, démarche, résultats et métriques, instructions                                                         d'installation et d'exécution, captures d'écran de l'application.  
*  Lien vers Jira

## **7\. Critères de performance**

* Qualité des Modèles:  
  * Précision des prédictions : Les modèles doivent atteindre des métriques de performance élevées sur l'ensemble de test.  
  * Tuning des hyperparamètres : Optimiser les hyperparamètres pour améliorer les performances du modèle.  
  * Robustesse et généralisation : Les modèles doivent être robustes et généraliser bien sur des données inédites.  
* Documentation et Reproductibilité:  
  * Qualité de la documentation : Fournir une documentation complète et claire décrivant chaque étape du projet.  
* Efficacité du Code:  
  * Lisibilité et organisation : Produire un code clair, bien structuré, et facile à maintenir.  
  * Performance computationnelle : Optimiser l'efficacité du code pour réduire le temps d'exécution et l'utilisation des ressources.  
  * Gestion des erreurs et robustesse : Produire un code robuste capable de gérer les erreurs et les inputs incorrects.