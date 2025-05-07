# multimodel-image-classification

![Langage](https://img.shields.io/badge/langage-Python-blue.svg)  

## Description

Ce projet propose une **pipeline complète** de classification binaire d’images (mer vs non-mer) reposant sur :  
1. Plusieurs **extractions de caractéristiques** (pixels bruts, canal bleu, histogrammes, HOG, LBP, Haralick, GLCM).  
2. Plusieurs **classifieurs** standards (Random Forest, Decision Tree, k-Nearest Neighbors, Naïve Bayes).  
3. Évaluation par **train/test split** et **cross-validation**.

L’objectif est de comparer automatiquement les combinaisons « feature + modèle » et de sélectionner la meilleure approche.

 **Collaboration :** projet réalisé en binôme dans le cadre du cours de L3 : « Introduction à l'apprentissage automatique ».
