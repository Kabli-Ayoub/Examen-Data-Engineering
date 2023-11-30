# Examen-Data-Engineering
Author: Mohamed & Ayoub & godwin 

## Table des Matières

1. [Introduction](#introduction)
2. [Approche Tandem](#)
3. [Docker](#)

---

## Introduction

Projet de Clustering Textuel avec Réduction de Dimension
Objectif
L'objectif de ce projet est de simuler un environnement de développement collaboratif où plusieurs développeurs travaillent sur un modèle de clustering textuel basé sur la réduction de dimensionnalité. Chaque développeur est responsable d'une approche spécifique (ACP, AFC, UMAP) et doit fusionner sa version finale avec la branche principale (main). Ensuite, un conteneur Docker sera créé à partir du repository GitHub (branche main) et se mettra automatiquement à jour en production.

Consignes
Repository GitHub
Le repository GitHub MLDS-AF/Examen contient des templates pour vous guider tout au long du processus.

Développement du Modèle
Développez un modèle de clustering basé sur la réduction de dimensionnalité (ACP, AFC, UMAP).
Utilisez les données textuelles NG20 fournies dans le template (utilisation de 2000 documents seulement).
La méthode tandem ou séquentielle combine la réduction de dimension avec l'algorithme de clustering K-means.
Créez un repository GitHub avec un README, un .gitignore, et ajoutez un fichier main.py évaluant chaque approche (ACP+kmeans, AFC+kmeans, UMAP+kmeans) avec les métriques NMI, ARI et Accuracy.
Développement Collaboratif
Travailler de manière indépendante et asynchrone en développant une approche par personne dans une nouvelle branche.
Utilisez le notebook template_branche.ipynb pour développer et tester le modèle.
Fusionnez votre branche avec la branche main via un merge pull request, en ajoutant votre notebook dans un dossier "experiments" et en mettant à jour le main.py avec votre approche.
Création de l'Image Docker
Clonez la branche main finale sur une machine.
Créez une image Docker pour exécuter le fichier main.py et retourner les résultats de clustering de chaque méthode.
Push vers Docker Hub
Poussez l'image créée dans Docker Hub.

## Docker 

lien vers l'image 
```shell

```