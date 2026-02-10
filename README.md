# `pipeline.py` (DDPM + ScoreNet/SGM) — Cómo ejecutar por terminal

Este repo contiene un script **secuencial** (`pipeline.py`) que ejecuta el pipeline completo:

- Carga **BCI Competition III – Dataset V** (Subject 1: `raw01+raw02+raw03`)
- Preprocesa (paper-like): **8–30 Hz bandpass + muscle annotation + ICA (Fp1/Fp2) + z-score**
- Segmenta epochs de 1s (512) con stride 0.5s (256), sin cruzar labels ni sesiones
- Split **70/30 estratificado**
- Entrena y evalúa:
  - **DDPM** (por canal target) + tabla MSE/Pearson
  - **CSP+LDA** (real vs híbrido)
  - **U-Net classifier** (paper hyperparams)
  - **ScoreNet/SGM** (por canal) + tabla + CSP+LDA (híbrido)

> Nota: el script **no genera imágenes/plots**; solo imprime **logs** y **tablas**.

---

## Requisitos de datos (carpeta `dataset/`)

`pipeline.py` asume que ejecutas desde la carpeta `Syntetic_functions/` y que existen estos archivos en la carpeta `Syntetic_functions/dataset/`:

```text
dataset/BCI_3/train_subject1_raw01.mat
dataset/BCI_3/train_subject1_raw02.mat
dataset/BCI_3/train_subject1_raw03.mat
```

Si no están, el script fallará al cargar los `.mat`.

1. Para ello ir a: https://www.bbci.de/competition/iii/download/index.html?agree=yes&submit=Submit

2. Descargar solo el subject 1 subject1 [ ASCII format (68 MB) ] [ Matlab format (45 MB) ], descargando solo el formato en .mat y posterior dejar los archivos .mat como `dataset/BCI_3/train_subject1_raw0X.mat/`:
3. 
---

## 1) Crear un conda environment

Recomendado: **Python 3.10 o 3.11** (por compatibilidad con `torch`, `mne`, etc.).

```bash
conda create -n eeg-diffusion python=3.11 -y
conda activate eeg-diffusion
```

---

## 2) Instalar dependencias

### Opción A (simple, con pip)

```bash
pip install -U pip
pip install numpy scipy pandas scikit-learn tqdm mne
```

Instala PyTorch según tu plataforma:

- CPU (suele funcionar en macOS/Linux/Windows):

```bash
pip install torch
```

- CUDA (si tienes GPU NVIDIA): sigue la guía oficial de PyTorch para tu versión de CUDA: `[https://pytorch.org/get-started/locally/]`

### Opción B (conda + conda-forge)

```bash
conda install -c conda-forge numpy scipy pandas scikit-learn tqdm mne -y
```

Luego instala PyTorch:

```bash
pip install torch
```

---

## 3) Ejecutar el script

Desde la carpeta del proyecto:

```bash
python pipeline.py
```
o simplemente ejecutalo en el idle o vscode.


- **Forzar device**:

```bash
python pipeline.py --device cpu
python pipeline.py --device cuda
```

---


