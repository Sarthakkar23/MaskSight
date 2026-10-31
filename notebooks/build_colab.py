"""Generates notebooks/colab_benchmark.ipynb (keeps the notebook diff-friendly to maintain)."""

import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()})


md("""
# MaskSight - full benchmark on a Colab GPU
Runs the main benchmark (2 methods x 15 MVTec AD categories), the ablation study, and fits the small
calibrated checkpoints used by the demo. Results are written incrementally, so if the runtime
disconnects just re-run all cells and finished categories are skipped.

**Runtime -> Change runtime type -> T4 GPU.**
""")
code("""
import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
""")
code("""
# Get the code: an uploaded /content/masksight.zip (Files panel) if present, else clone from GitHub
import os, subprocess
REPO = "https://github.com/Sarthakkar23/MaskSight.git"
if not os.path.exists("/content/MaskSight"):
    if os.path.exists("/content/masksight.zip"):
        subprocess.run(["unzip", "-q", "-o", "/content/masksight.zip", "-d", "/content/MaskSight"], check=True)
    else:
        subprocess.run(["git", "clone", "-q", REPO, "/content/MaskSight"], check=True)
%cd /content/MaskSight
!pip -q install -e ".[data]"
""")
code("""
!masksight download --root /content/mvtec
""")
code("""
# Main benchmark (~35 min on a T4)
!masksight benchmark configs/main.yaml
""")
code("""
# Ablation study on 5 categories (~35 min)
!masksight benchmark configs/ablation.yaml
""")
code("""
# Small calibrated checkpoints for the demo / API (1% coreset, fp16 memory bank, ~5 MB each)
from masksight.data import MVTEC_CATEGORIES
for c in MVTEC_CATEGORIES:
    if not os.path.exists(f"models/{c}.pt"):
        !masksight fit --category {c} --data-root /content/mvtec --out models/{c}.pt
""")
code("""
# A few demo example images (one defective + one good per category)
import shutil, glob
os.makedirs("app/examples", exist_ok=True)
for c in MVTEC_CATEGORIES:
    bad = sorted(p for p in glob.glob(f"/content/mvtec/{c}/test/*/*.png") if "/good/" not in p)
    shutil.copy(bad[len(bad) // 2], f"app/examples/{c}__defect.png")
    shutil.copy(sorted(glob.glob(f"/content/mvtec/{c}/test/good/*.png"))[0], f"app/examples/{c}__good.png")
""")
code("""
!masksight report results/main/results.csv --per-category
!masksight report results/ablation/results.csv
""")
code("""
# Bundle everything to bring back into the repo
!zip -q -r /content/masksight_outputs.zip results models app/examples
from google.colab import files; files.download("/content/masksight_outputs.zip")
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 0,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
Path(__file__).with_name("colab_benchmark.ipynb").write_text(json.dumps(nb, indent=1))
