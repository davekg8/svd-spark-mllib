# SVD avec Spark MLlib : partie application

Ce projet montre deux applications de la SVD, calculées avec l'API **RDD** de Spark
(`pyspark.mllib.linalg.distributed.IndexedRowMatrix.computeSVD`).

| Fichier | Contenu |
|---|---|
| `01_compression_image.py` / `.ipynb` | Démo : compression d'une image 512 × 512 par SVD tronquée (k = 5, 20, 50, 100) |
| `02_recommandation_movielens.py` / `.ipynb` | Application : recommandation de films sur MovieLens (RMSE, top 10, films similaires, facteurs latents) |
| `spark_utils.py` | Création de la SparkSession (corrections pour Windows) et mise en cache JVM de la matrice |
| `build_notebooks.py` | Régénère les notebooks à partir des scripts et les exécute |
| `figures/` | Graphiques utilisés dans les slides, avec `resultats_movielens.json` |
| `data/` | MovieLens `ml-latest-small`, téléchargé automatiquement si absent |

## Installation

Prérequis : Java 17 ou 21 (`java -version`) et Python 3.10 à 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
```

## Lancement

```bash
python 01_compression_image.py
python 02_recommandation_movielens.py
```

Pour ouvrir les notebooks : `jupyter lab`. Pour les régénérer avec leurs sorties : `python build_notebooks.py`.

## Résultats principaux

- **Compression :** avec k = 50, on stocke 20 % des valeurs pour une erreur relative de 8 %.
- **MovieLens :** sur 610 utilisateurs × 2 269 films (films notés ≥ 10 fois), split 80 / 20 :

| Modèle | RMSE test |
|---|---|
| Moyenne globale | 1,021 |
| Moyenne de l'utilisateur | 0,935 |
| μ + b_u + b_i | 0,866 |
| **+ SVD des résidus, k = 10** | **0,851** |

Au-delà de k ≈ 10, l'erreur remonte : il y a sur-apprentissage.

## Notes techniques

- La SVD n'existe que dans l'API RDD (`spark.mllib`). L'API Pipeline (`spark.ml`) ne propose que PCA.
- `computeSVD` utilise ARPACK, qui relit la matrice à chaque itération. `spark_utils.cache_jvm` met donc
  les lignes en cache côté JVM. Sans ce cache, le calcul est très lent, car les workers Python sont relancés à chaque passage.
- Sous Windows, si le nom de la machine contient un `_`, Spark échoue (« Invalid Spark URL »).
  `spark_utils` force donc `localhost`.

## Source des données

F. M. Harper & J. A. Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), 2015.
https://grouplens.org/datasets/movielens/
