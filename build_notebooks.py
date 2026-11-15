"""Convertit les scripts `# %%` en notebooks Jupyter et les exécute (sorties incluses).

Usage :  .venv\\Scripts\\python.exe build_notebooks.py
"""
import re
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

SCRIPTS = ["01_compression_image.py", "02_recommandation_movielens.py"]


def py_to_nb(path):
    text = open(path, encoding="utf-8").read()
    chunks = re.split(r"^# %%(.*)$", text, flags=re.M)
    cells = []
    # chunks = [avant, marqueur1, contenu1, marqueur2, contenu2, ...]
    for marker, body in zip(chunks[1::2], chunks[2::2]):
        body = body.strip("\n")
        if "[markdown]" in marker:
            md = "\n".join(re.sub(r"^# ?", "", l) for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(md))
        elif body.strip():
            cells.append(nbformat.v4.new_code_cell(body))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    for script in SCRIPTS:
        nb = py_to_nb(script)
        out = script.replace(".py", ".ipynb")
        print(f"Exécution de {out} ...", flush=True)
        ExecutePreprocessor(timeout=1200, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
        nbformat.write(nb, out)
        print(f"  -> {out} écrit", flush=True)
    sys.exit(0)
