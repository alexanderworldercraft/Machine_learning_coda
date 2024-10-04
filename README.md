# Machine_learning_coda

![IA](./asset/ia.webp "Image illustrant l'IA.")

## Sommaire :

- [**Jour 1️⃣ : Grands concepts, machine learning supervisé, introduction aux outils.**](#jour-1️⃣--grands-concepts-machine-learning-supervisé-introduction-aux-outils)
  - [Qu'est ce que le machine learning ?](#quest-ce-que-le-machine-learning-)
  - [kaggle](#kaggle)
    - [Qu'est ce que le site kaggle ?](#quest-ce-que-le-site-keggle-)
- [**Jour 2️⃣ : Machine learning non supervisé, corrélation et causalité, explicabilité.**](#jour-2️⃣--machine-learning-non-supervisé-corrélation-et-causalité-explicabilité)
  - [Exemple de modèles avancés](#exemple-de-modèles-avancés)
  - [les Forêts Aléatoires](#les-forêts-aléatoires)
    - [Comment fonctionnent les Forêts Aléatoires ?](#comment-fonctionnent-les-forêts-aléatoires-)
    - [Avantages des Forêts Aléatoires](#avantages-des-forêts-aléatoires)
    - [Limitations des Forêts Aléatoires](#limitations-des-forêts-aléatoires)
    - [Applications des Forêts Aléatoires](#applications-des-forêts-aléatoires)
    - [Pourquoi les Forêts Aléatoires sont-elles efficaces ?](#pourquoi-les-forêts-aléatoires-sont-elles-efficaces-)
  - [le Boosting]








- [**Jour 3️⃣ : Deep learning, vision par ordinateur, traîtement du langage naturel.**](#jour-3️⃣--deep-learning-vision-par-ordinateur-traîtement-du-langage-naturel)
- [**Jour 4️⃣ : LLMs et IA Générative**](#jour-4️⃣--llms-et-ia-générative)
- [**jour 5️⃣ : Les métiers de la Data et de l’IA, IA & éthique, conclusion du challenge**](#jour-5️⃣--les-métiers-de-la-data-et-de-lia-ia--éthique-conclusion-du-challenge)

- [**🤖 Challenge**]()
  - [Prédire la vitesse d'adoption d'animaux de compagnie en fonction de l'annonce postée.]()
  - [Recommander aux sauveteurs d'animaux des stratégies pour optimiser les annnonces d'adoption.]()
  - [Fournir un code que les équipes vont pouvoir reprendre, industrialiser et améliorer avec moins d'effort possible.]()

- [**🕵️‍♂️ Liens utile**](#liens-utile)






- [**🏁 Conclusion**](#conclusion)
- [**🥇 Contribution**](#contribution)
  - [👦 Contribueur](#contribueur)
- [**📄 Licence**](#licence)

## Jour 1️⃣ : Grands concepts, Machine Learning supervisé, introduction aux outils

### Qu'est ce que le machine learning ?

Le **Machine Learning (apprentissage automatique)** est un sous-domaine de l'intelligence artificielle (IA) qui consiste à créer des systèmes capables d'apprendre à partir de données sans être explicitement programmés pour une tâche spécifique. En d'autres termes, au lieu de coder un programme pour résoudre un problème, on fournit à l'algorithme des exemples (données d'entraînement), et il apprend à reconnaître des motifs ou des régularités dans ces données pour ensuite faire des prédictions ou des décisions sur des données nouvelles.

Il existe trois types principaux d'apprentissage automatique :

1. **Apprentissage supervisé** : L'algorithme apprend à partir d'exemples étiquetés (`données avec la réponse correcte`). Ex : reconnaissance d'images, prédiction de prix. (Ne change pas de comportement).

2. **Apprentissage non supervisé** : L'algorithme essaie de trouver des structures ou des motifs dans des données `non étiquetées`. Ex : clustering, réduction de dimensionnalité. (Peux changé de comportement).

3. **Apprentissage par renforcement** : L'algorithme apprend à `prendre des décisions en interagissant avec un environnement` et en recevant des récompenses ou des punitions en fonction de ses actions. Ex : jeux vidéo (**vidéo IA sur Track mania**), robots. (Commence sans données.)

Le machine learning est utilisé dans divers domaines comme la reconnaissance d'image, la recommandation de produits, le traitement du langage naturel, et bien plus encore.


## kaggle

lien du site [ici](https://www.kaggle.com/).

### Qu'est ce que le site Keggle ?

Kaggle est une plateforme en ligne spécialisée dans le **data science** et le **machine learning**. Voici ses principales fonctionnalités :

1. **Compétitions** : Kaggle est surtout connu pour ses compétitions de machine learning, où des entreprises et des chercheurs publient des ensembles de données et des problèmes à résoudre. Les participants soumettent leurs modèles, et les meilleurs modèles sont souvent récompensés par des prix en argent.

2. **Datasets** : Kaggle fournit une vaste collection de jeux de données gratuits que les utilisateurs peuvent explorer, analyser et utiliser pour leurs projets personnels ou professionnels.

3. **Kernels (Notebooks)** : Les utilisateurs peuvent créer des notebooks (souvent en Python ou R) directement sur la plateforme pour effectuer des analyses de données, entraîner des modèles, et visualiser des résultats. Il n'est pas nécessaire de télécharger les jeux de données sur son propre ordinateur, tout peut se faire en ligne.

4. **Apprentissage** : Kaggle propose des tutoriels et des cours gratuits pour apprendre des compétences en data science, en machine learning et en intelligence artificielle, allant des bases à des techniques avancées.

5. **Communauté** : La plateforme abrite une grande communauté d'experts en data science, ce qui permet aux utilisateurs de poser des questions, partager leurs idées, et collaborer sur des projets.

En résumé, Kaggle est un outil puissant pour tous ceux qui s'intéressent à la **science des données**, du débutant au professionnel.


## Jour 2️⃣ : Machine Learning non supervisé, corrélation et causalité, explicabilité
### Exemple de modèles avancés
  - [modèles avancés du prof](./asset/Correction%20Modèles%20Avancés.ipynb)
### les Forêts Aléatoires

**Les Forêts Aléatoires** (en anglais *Random Forests*) sont une méthode d'apprentissage automatique qui combine plusieurs arbres de décision pour améliorer la précision des prédictions et réduire le risque de surapprentissage (*overfitting*). Voici une explication plus détaillée de leur fonctionnement et de leurs avantages.

#### Comment fonctionnent les Forêts Aléatoires ?
1. **Construction de multiples arbres de décision** : Une Forêt Aléatoire est composée de nombreux arbres de décision individuels. Chaque arbre est construit à partir d'un échantillon différent des données d'entraînement.
2. **Échantillonnage avec remise (Bootstrap)** : Pour chaque arbre, un sous-ensemble aléatoire des données d'entraînement est sélectionné avec remise. Cela signifie que certains échantillons peuvent être choisis plusieurs fois pour un arbre donné, tandis que d'autres peuvent ne pas être sélectionnés.
3. **Sélection aléatoire des caractéristiques** : Lors de la construction de chaque arbre, à chaque nœud, un sous-ensemble aléatoire de caractéristiques est considéré pour déterminer la meilleure division. Cette technique ajoute de la diversité entre les arbres.
4. **Agrégation des prédictions** :
    - **Pour la classification** : Chaque arbre "vote" pour une classe, et la classe la plus votée est choisie comme prédiction finale.
    - **Pour la régression** : La moyenne des prédictions de tous les arbres est calculée pour obtenir la prédiction finale.
#### Avantages des Forêts Aléatoires
- **Amélioration de la précision** : En combinant les prédictions de multiples arbres, les Forêts Aléatoires atteignent généralement une meilleure performance que les arbres de décision individuels.
- **Réduction du surapprentissage** : Les arbres de décision peuvent facilement surapprendre les données d'entraînement. Les Forêts Aléatoires atténuent ce problème grâce à la diversité introduite par l'échantillonnage aléatoire des données et des caractéristiques.
- **Robustesse aux données bruitées** : Elles sont moins sensibles aux anomalies et aux valeurs aberrantes dans les données.
- **Gestion efficace des grandes bases de données** : Elles peuvent gérer efficacement un grand nombre de caractéristiques et d'échantillons sans nécessiter beaucoup de prétraitement des données.
- **Estimation de l'importance des variables** : Les Forêts Aléatoires peuvent fournir une mesure de l'importance de chaque caractéristique dans la prédiction, aidant à identifier les variables les plus influentes.
#### Limitations des Forêts Aléatoires
- **Perte d'interprétabilité** : Contrairement à un arbre de décision unique, les Forêts Aléatoires sont plus difficiles à interpréter en raison du grand nombre d'arbres impliqués.
- **Temps de calcul plus long** : Entraîner plusieurs arbres et agréger leurs prédictions peut être plus gourmand en temps et en ressources informatiques.
- **Possibilité de biais** : Si les données contiennent des biais, les Forêts Aléatoires peuvent les reproduire, car elles n'ont pas de mécanisme intrinsèque pour les corriger.
#### Applications des Forêts Aléatoires
Les Forêts Aléatoires sont utilisées dans de nombreux domaines en raison de leur efficacité et de leur flexibilité :
- **Médecine** : Diagnostic de maladies, analyse génétique.
- **Finance** : Prédiction de risques, détection de fraudes.
- **Marketing** : Segmentation de clientèle, prédiction du comportement des consommateurs.
- **Environnement** : Prévision météorologique, modélisation écologique.
- **Informatique** : Reconnaissance d'images, traitement du langage naturel.
#### Pourquoi les Forêts Aléatoires sont-elles efficaces ?
- **Diversification des modèles** : En entraînant chaque arbre sur des données et des caractéristiques différentes, les Forêts Aléatoires créent une collection d'arbres variés qui, ensemble, capturent une image plus complète des données.
- **Réduction de la variance** : La combinaison des prédictions de plusieurs arbres réduit la variance globale du modèle, conduisant à des prédictions plus stables et fiables.
- **Principe de la "sagesse des foules"** : Comme un groupe d'experts peut prendre une meilleure décision qu'un seul individu, une Forêt Aléatoire utilise la sagesse collective de nombreux arbres pour améliorer la précision.
----
### le Boosting
Les **algorithmes de Boosting** sont une famille de méthodes d'apprentissage automatique qui visent à améliorer la précision des prédictions en combinant plusieurs modèles simples (appelés apprenants faibles) pour créer un modèle puissant. L'idée fondamentale du Boosting est de construire un ensemble de modèles qui corrigent successivement les erreurs des précédents.
#### Comment fonctionne le Boosting ?
1. **Initialisation :**
    - On commence par entraîner un modèle de base sur l'ensemble des données d'entraînement.
    - Ce modèle peut être un arbre de décision peu profond, appelé "souche" (stump).
2. **Pondération des observations :**
    - Après l'entraînement du premier modèle, on évalue ses performances.
    - Les observations mal prédites reçoivent un poids plus élevé, ce qui signifie qu'elles seront davantage prises en compte lors de l'entraînement du prochain modèle.
3. **Itération :**
    - Un nouveau modèle est entraîné, en se concentrant sur les observations qui ont été mal prédites par le modèle précédent.
    - Ce processus est répété plusieurs fois, chaque nouveau modèle tentant de corriger les erreurs des modèles antérieurs.
4. **Combinaison des modèles :**
    - Les prédictions finales sont obtenues en combinant les prédictions de tous les modèles.
    - Pour la classification, cela peut se faire par un vote pondéré ; pour la régression, par une moyenne pondérée.
#### Types d'algorithmes de Boosting
1. **AdaBoost (Adaptive Boosting) :**
    - **Principe** : Ajuste les poids des observations à chaque itération en accordant plus d'importance aux erreurs.








## Jour 3️⃣ : Deep Learning, vision par ordinateur, traîtement du langage naturel
## Jour 4️⃣ : LLMs et IA Générative
## Jour 5️⃣ : Les métiers de la Data et de l’IA, IA & éthique, conclusion du challenge
## Challenge
### Prédire la vitesse d'adoption d'animaux de compagnie en fonction de l'annonce postée.
### Recommander aux sauveteurs d'animaux des stratégies pour optimiser les annnonces d'adoption.
### Fournir un code que les équipes vont pouvoir reprendre, industrialiser et améliorer avec moins d'effort possible.

# Liens utile

Liens des slides :
- [https://docs.google.com/presentation/d/1PfxoBpOC2gWMZONN5JIzRF1WzBcfEyEpeTJcHPQJc6Y/edit?usp=sharing](https://docs.google.com/presentation/d/1PfxoBpOC2gWMZONN5JIzRF1WzBcfEyEpeTJcHPQJc6Y/edit?usp=sharing) ou [`ici`](./asset/Introduction%20au%20Machine%20Learning.pdf)
- [https://docs.google.com/presentation/d/1Mdtt6g66YPeiun7eoobayZSmj5bxmX_fOWvSEaxGkeU/edit?usp=sharing](https://docs.google.com/presentation/d/1Mdtt6g66YPeiun7eoobayZSmj5bxmX_fOWvSEaxGkeU/edit?usp=sharing) ou [`ici`](./asset/Introduction.pdf)
- [https://docs.google.com/presentation/d/14fgS2g6At8xmurBJhkfYC_XaK2xDil2G5cGa6GOHUPw/edit#slide=id.p1](https://docs.google.com/presentation/d/14fgS2g6At8xmurBJhkfYC_XaK2xDil2G5cGa6GOHUPw/edit#slide=id.p1) ou [`ici`](./asset/Deep%20Learning%20et%20vision%20par%20ordinateur.pdf)

Liens des tutoriels Kaggle :
- [https://www.kaggle.com/learn/python](https://www.kaggle.com/learn/python)
- [https://www.kaggle.com/learn/pandas](https://www.kaggle.com/learn/pandas)
- [https://www.kaggle.com/learn/intro-to-machine-learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- [https://www.kaggle.com/learn/data-visualization](https://www.kaggle.com/learn/data-visualization)
- [https://www.kaggle.com/learn/data-cleaning](https://www.kaggle.com/learn/data-cleaning)
- [https://www.kaggle.com/learn/intermediate-machine-learning](https://www.kaggle.com/learn/intermediate-machine-learning)
- [https://www.kaggle.com/learn/feature-engineering](https://www.kaggle.com/learn/feature-engineering)

Lien du répo github : [https://github.com/vlandeau/data_dojo/tree/master](https://github.com/vlandeau/data_dojo/tree/master)

Lien de Google Colab : [https://colab.research.google.com/](https://colab.research.google.com/)

Lien du challenge : [https://www.kaggle.com/t/a893379e9b624b308e5151d9ab7ed1f2](https://www.kaggle.com/t/a893379e9b624b308e5151d9ab7ed1f2)

Lien d'explication des modèles : slide [https://docs.google.com/presentation/d/1AfYnGmOwGxwM4DJsmhzP4ucz9LQ9IJFmwkh8_Mx4W94/edit#slide=id.g1879f72834b_0_1198](https://docs.google.com/presentation/d/1AfYnGmOwGxwM4DJsmhzP4ucz9LQ9IJFmwkh8_Mx4W94/edit#slide=id.g1879f72834b_0_1198) ou [ici](/asset/Shapash.pdf)

Lien d'explication des modèles : colab [https://colab.research.google.com/drive/1D9-o6Nouv1M2NA8LDV3CFDORqHr1LoSd?usp=sharing](https://colab.research.google.com/drive/1D9-o6Nouv1M2NA8LDV3CFDORqHr1LoSd?usp=sharing)

Lien des features catégorielles : [https://colab.research.google.com/drive/1eC9P_RGowpnSjfurUy-K0UwoUU9PUHRp?usp=sharing](https://colab.research.google.com/drive/1eC9P_RGowpnSjfurUy-K0UwoUU9PUHRp?usp=sharing)

Librairies utilisées :
- [Shapash](https://github.com/MAIF/shapash)
- [Shap](https://shap.readthedocs.io/en/latest/index.html)
- [category_encoders](https://contrib.scikit-learn.org/category_encoders/)

## 🏁 Conclusion

## Contribution

Les contributions à ce repository sont les bienvenues ! Si vous souhaitez corriger une erreur ou améliorer le contenu existant, n'hésitez pas à m'en faire part.

### Contribueur

- [**👨‍💻🥇 Alexander worldercraft**](https://github.com/alexanderworldercraft)

## Licence

Ce contenu est sous licence [GNU GPLv3](LICENSE.txt). Vous êtes libre de l'utiliser, le modifier et le distribuer selon les termes de cette licence.