# 🩺 Détection de Pneumonie par Radiographie Thoracique

> Projet académique — **ENSA Khouribga**, Filière IID1, Année 2025–2026  
> Module : Méthodologie de projet et veille scientifique

**Réalisé par :** Deroui Mouad · Laaouina Othmane · Maktou Azeddine · Nougbolo Godwin Elie  
**Encadré par :** Pr. Safiny Yazid · Pr. Siham Ahmam

Classification binaire (**NORMAL / PNEUMONIA**) à partir de radiographies pulmonaires, via deux approches Deep Learning : un CNN personnalisé et le Transfer Learning avec DenseNet121.

---

## 📋 Table des matières

- [Contexte médical](#contexte-médical)
- [Problématique](#problématique)
- [État de l'art](#état-de-lart)
- [Dataset](#dataset)
- [Structure du projet](#structure-du-projet)
- [Modèles](#modèles)
- [Résultats](#résultats)
- [Application Streamlit](#application-streamlit)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Technologies utilisées](#technologies-utilisées)
- [Auteurs](#auteurs)

---

## Contexte médical

La pneumonie est une infection pulmonaire grave qui tue chaque année plusieurs millions de personnes, en particulier des enfants de moins de cinq ans et des personnes âgées. Son diagnostic repose sur l'analyse visuelle de radiographies thoraciques — une tâche longue, exigeant de l'expertise, et difficilement scalable dans les régions à ressources médicales limitées.

Sur une radiographie **normale**, les champs pulmonaires apparaissent sombres (remplis d'air). En présence d'une **pneumonie**, certaines zones deviennent opaques ou grisâtres, traduisant une accumulation de liquide ou une infection localisée.

L'intelligence artificielle offre une réponse partielle à ce problème : les CNN ont démontré des performances comparables à celles de radiologues expérimentés sur des tâches de classification d'images médicales.

---

## Problématique

> **Comment concevoir un système de Deep Learning capable de classifier automatiquement une radiographie thoracique en NORMAL ou PNEUMONIA, avec un recall suffisamment élevé pour limiter les faux négatifs dans un contexte médical ?**

### Difficultés principales

| Défi | Description |
|------|-------------|
| **Déséquilibre des classes** | La classe PNEUMONIA représente ~74% des images d'entraînement, ce qui peut biaiser l'apprentissage |
| **Variabilité des images** | Dimensions très variables (largeur : 384–2 916 px, hauteur : 127–2 663 px) |
| **Overfitting** | Dataset médical de taille modérée → risque d'apprentissage par cœur |
| **Faux négatifs** | Un cas de pneumonie classé NORMAL peut retarder la prise en charge avec des conséquences graves |

> **Priorité métrique : le Recall sur la classe PNEUMONIA**, plus que l'accuracy globale.

---

## État de l'art

Quatre grandes approches ont été analysées dans la littérature :

| Approche | Avantages | Limites |
|----------|-----------|---------|
| **Machine Learning classique** (SVM, RF) | Interprétable, peu coûteux | Extraction manuelle de features, performances limitées |
| **CNN with Transfer Learning** | Très bonnes performances, peu de données nécessaires | Boîte noire, coût computationnel |
| **Ensemble Learning** | Robustesse | Complexité d'implémentation |
| **Vision Transformers (ViT)** | État de l'art sur grands datasets | Nécessite énormément de données |

**Positionnement de ce projet :** CNN from scratch vs Transfer Learning avec DenseNet121 — comparaison rigoureuse sur le même dataset, avec optimisation du seuil de décision et déploiement applicatif.

---

## Dataset

Le dataset utilisé est le **[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** disponible sur Kaggle.

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | ~1 341 | ~3 875 | ~5 216 |
| Validation | 20% du train (split automatique) | | |
| Test | 234 | 390 | 624 |

**Classes :**
- `NORMAL` — poumons sains
- `PNEUMONIA` — poumons infectés (bactérienne ou virale)

**Pré-traitement :**
- Redimensionnement des images : **224 × 224 px**
- Normalisation des pixels : `[0, 1]`
- Split validation : **20%** du training set (`validation_split=0.2`)

---

## Structure du projet

```
📦 Pneumonia_Project/
├── 📁 app/
│   └── app.py                        # Application Streamlit
├── 📁 models/
│   ├── 🧠 pneumonia_cnn_model.keras  # Modèle CNN v1
│   ├── 🧠 cnn_new.keras              # Modèle CNN v2 (optimisé)
│   ├── 🧠 densenet121_model.keras    # Modèle DenseNet121 v1
│   └── 🧠 densenet_new.keras         # Modèle DenseNet121 v2 (optimisé)
├── 📁 notebooks/
│   ├── 📓 CNN_Pneumonia.ipynb        # Notebook CNN personnalisé
│   └── 📓 DenseNet121_Pneumonia.ipynb # Notebook Transfer Learning DenseNet121
├── 📁 assets/                        # Ressources visuelles
├── convert_models.py                 # Script de conversion des modèles
├── requirements.txt                  # Dépendances Python
├── packages.txt                      # Paquets système
└── README.md
```

---

## Modèles

### 1. CNN Personnalisé (`CNN_Pneumonia.ipynb`)

Architecture construite **from scratch** :

- **Input** : `(224, 224, 3)`
- **Data Augmentation** : RandomFlip, RandomRotation (0.02), RandomZoom (0.1), RandomTranslation
- **Blocs CNN** : plusieurs couches `Conv2D` + `MaxPooling2D` + `Dropout`
- **Tête de classification** : `GlobalAveragePooling2D` → `Dense` → sortie sigmoïde
- **Optimizer** : Adam (`lr = 0.0001`)
- **Loss** : Binary Crossentropy
- **Métriques** : Accuracy, Precision, Recall, AUC
- **Callbacks** : EarlyStopping (patience=5), ModelCheckpoint
- **Epochs max** : 25 | **Batch size** : 32 | **Arrêt réel** : 13 epochs
- **Threshold optimal** : **0.7**

---

### 2. DenseNet121 — Transfer Learning (`DenseNet121_Pneumonia.ipynb`)

Approche en **deux phases** :

**Phase 1 — Feature Extraction**
- Base DenseNet121 pré-entraînée sur ImageNet, couches gelées (`trainable=False`)
- Tête : `GlobalAveragePooling2D` → `Dense` → `Dropout` → sortie sigmoïde
- Optimizer : Adam (`lr = 0.0001`) | Epochs : 15

**Phase 2 — Fine Tuning**
- Dégel des dernières couches de la base DenseNet121
- Réentraînement avec learning rate réduit | Epochs : 10

**Data Augmentation** : RandomFlip, RandomRotation (0.05), RandomZoom, RandomContrast  
**Threshold optimal** : **0.8**

> **Résultat clé :** La phase de Fine Tuning n'a pas amélioré les résultats de la Feature Extraction — DenseNet121 est suffisamment puissant tel quel pour cette tâche.

---

## Résultats

### CNN Personnalisé — Test Set (threshold = 0.7)

| Métrique | NORMAL | PNEUMONIA | Global |
|----------|--------|-----------|--------|
| Precision | 0.72 | 0.88 | 0.82 |
| Recall | 0.81 | 0.81 | 0.81 |
| F1-score | 0.76 | 0.84 | 0.81 |
| **Accuracy** | | | **81%** |
| **F1 macro** | | | **0.8415** |

---

### DenseNet121 — Feature Extraction — Test Set (threshold = 0.8)

| Métrique | NORMAL | PNEUMONIA | Global |
|----------|--------|-----------|--------|
| Precision | 0.80 | 0.93 | 0.89 |
| Recall | 0.90 | 0.87 | 0.88 |
| F1-score | 0.85 | 0.90 | 0.88 |
| **Accuracy** | | | **88.9%** |
| **F1 macro** | | | **0.9109** |

---

### Comparaison finale

| Modèle | Accuracy | F1 (PNEUMONIA) | Recall (PNEUMONIA) | AUC |
|--------|----------|----------------|--------------------|-----|
| CNN custom | 80.3% | 0.84 | 0.81 | ~0.89 |
| DenseNet121 (FE) | **88.9%** | **0.91** | **0.92** | **~0.95** |
| DenseNet121 (FT) | ≈ FE | ≈ FE | ≈ FE | ≈ FE |

> **DenseNet121 surpasse le CNN personnalisé** sur toutes les métriques grâce au Transfer Learning depuis ImageNet. Le Fine Tuning n'apporte pas de gain supplémentaire significatif.

### Comparaison avec la littérature

Les résultats de DenseNet121 (accuracy 88.9%, recall PNEUMONIA 0.92) sont cohérents avec l'état de l'art sur ce dataset. Ils s'approchent des performances rapportées dans les meilleures études sans recours à des architectures plus lourdes (VGG, ResNet-50, ViT).

---

## Application Streamlit

Une interface web interactive permet de tester les deux modèles en temps réel :

- Upload d'une radiographie (JPEG/PNG)
- Sélection du modèle (CNN ou DenseNet121)
- Affichage du diagnostic (NORMAL / PNEUMONIA) et du score de confiance
- Visualisation des probabilités

```bash
streamlit run app/app.py
```

---

## Installation

### Prérequis

- Python 3.9+
- GPU recommandé pour l'entraînement (notebooks conçus pour Google Colab)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

**`requirements.txt` :**
```
streamlit
tensorflow
numpy
matplotlib
plotly
pillow
```

### Dataset

Télécharger depuis Kaggle :
```
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```

Structure attendue :
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Utilisation

### Lancer l'application

```bash
streamlit run app/app.py
```

### Entraînement (Google Colab recommandé)

```
CNN_Pneumonia.ipynb          → CNN from scratch
DenseNet121_Pneumonia.ipynb  → Transfer Learning DenseNet121
```

### Chargement d'un modèle sauvegardé

```python
from tensorflow.keras.models import load_model

model = load_model("models/cnn_new.keras")           # CNN
model = load_model("models/densenet_new.keras")      # DenseNet121
```

### Prédiction sur une image

```python
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

def predict_image(model, image_path, threshold=0.7):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prediction >= threshold else "NORMAL"

    return label, float(prediction)

label, score = predict_image(model, "chest_xray.jpg")
print(f"Diagnostic : {label} (score : {score:.4f})")
```

---

## Technologies utilisées

| Technologie | Usage |
|-------------|-------|
| TensorFlow / Keras | Construction et entraînement des modèles |
| DenseNet121 | Transfer Learning (ImageNet) |
| NumPy | Manipulation des arrays |
| Matplotlib | Visualisation des courbes d'entraînement |
| Plotly | Graphiques interactifs |
| Pillow (PIL) | Chargement et traitement des images |
| Streamlit | Interface de démonstration web |
| Scikit-learn | Métriques (classification report, matrice de confusion) |
| Google Colab | Environnement d'entraînement (GPU) |

---

## Auteurs

Projet réalisé dans le cadre du module **Méthodologie de Projet et Veille Scientifique**, filière **IID1**, à l'**ENSA Khouribga** — Université Sultan Moulay Slimane.

**Réalisé par :**
- Deroui Mouad
- Laaouina Othmane
- Maktou Azeddine
- Nougbolo Godwin Elie

**Encadré par :**
- Pr. Safiny Yazid (encadrement technique)
- Pr. Siham Ahmam (cours et méthodologie)

---

*Année universitaire 2025–2026*
