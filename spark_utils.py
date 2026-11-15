"""Démarrage d'une SparkSession locale qui fonctionne sous Windows.

- SPARK_LOCAL_HOSTNAME / spark.driver.host : évite l'erreur « Invalid Spark URL »
  quand le nom de la machine contient un underscore (ex. PC_Dave).
- PYSPARK_PYTHON : les workers utilisent le même Python que le driver (le venv).
"""
import os
import sys
import warnings

os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
warnings.filterwarnings("ignore", category=FutureWarning)

from pyspark.sql import SparkSession  # noqa: E402


def get_spark(app_name="SVD-application", cores=4):
    spark = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def cache_jvm(matrix):
    """Met en cache (côté JVM) les lignes d'une RowMatrix / IndexedRowMatrix.

    computeSVD est itératif (ARPACK) : il relit la matrice des dizaines de fois.
    Sans cache, chaque passage relance les workers Python qui ont créé le RDD,
    ce qui est très lent. On matérialise donc les lignes une fois dans la JVM.
    """
    jrows = matrix._java_matrix_wrapper._java_model.rows()
    jrows.cache()
    jrows.count()
    return matrix
