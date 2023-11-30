# Examen-Data-Engineering
Author: **Mohamed & Ayoub & Godwin**

## Table des Matières

1. [Introduction](#introduction)
2. [Structure du projet](#structure-du-projet)
2. [Guide de développement](#guide-de-développement)
3. [Docker](#)

---
## Introduction
Ce projet simule un environnement de développement collaboratif où plusieurs développeurs travaillent sur un modèle de clustering textuel basé sur la réduction de dimensionnalité. Chaque développeur est responsable d'une approche spécifique (ACP, AFC, UMAP) et doit fusionner sa version finale avec la branche principale (main). Ensuite, un conteneur Docker sera créé à partir du repository GitHub (branche main) et se mettra automatiquement à jour en production.

## Structure du Projet
- experiments: Ce dossier contient les notebooks de chaque développeur, présentant leurs approches spécifiques.

- main.py: Fichier principal pour l'évaluation des différentes approches (ACP+kmeans, TSE+kmeans, UMAP+kmeans) à l'aide des métriques NMI, ARI et Accuracy.

- data: Contient les données textuelles NG20 utilisées dans le projet.

## Guide de Développement
- Cloner le Repository:

```bash
git clone git@github.com:Kabli-Ayoub/Examen-Data-Engineering.git
```


## Partie Docker

L'image docker se situe en: 
```bash
$ docker pull goamegah/examde
```
Pour construire l'image, utiliser la commande suivante:

```bash
$ docker build -t <image_exam_new> .
```

Pour lancer le conteneur, utiliser la commande suivante:

```bash
$ docker run -name <image_exam_cont> -p 5000:5000 image_exam_new
```
expected output:

```bash
Average NMI on 2 iterations for ACP: 0.42
Average ARI on 2 iterations for ACP: 0.24 

Average NMI on 2 iterations for TSNE: 0.45
Average ARI on 2 iterations for TSNE: 0.24 

Average NMI on 2 iterations for UMAP: 0.48
Average ARI on 2 iterations for UMAP: 0.24 
```