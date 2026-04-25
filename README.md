# CIST
# CIST: Clade-Informed Sequence Transformer for Influenza Antigenicity Prediction

This repository contains the code and data associated with the manuscript:

> **CIST: A Clade-Informed Sequence Transformer Framework for Predicting Influenza Virus Antigenicity**
> Kang Hu, Yongshan Zhu, Qingchuan Zhang\*, Jian Li\*
> *Scientific Reports* (under review)

---

## Requirements

```
Python 3.8
PyTorch 1.7.0
scikit-learn 1.3.2
xgboost 1.7.6
transformers 4.12.0
numpy 1.21.0
pandas 1.3.0
scipy 1.7.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

**NT dataset:** 837 virus–vaccine paired samples (93 H3N2 epidemic strains × 9 WHO-recommended vaccine strains, 2011–2020). GISAID accession numbers and clade labels are provided in `data/nt_dataset/`. Raw neutralizing antibody values are not held by the current authors; requests may be directed to the corresponding author of Zhang et al. (2022, *eBioMedicine*) at wangyc@nifdc.org.cn.

**HI dataset:** 12,768 virus–antiserum pairs (266 H3N2 strains × 48 antisera, 1968–2022). Processed HI titer data are provided in `data/hi_dataset/`. Original data sourced from Koel et al. (2013) and Worldwide Influenza Centre annual reports, integrated by Li et al. (2024, *Front. Microbiol.*).

---

## Reproducibility

All models were trained with fixed random seeds (`torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)`) under a 10-fold cross-validation protocol with strain-level stratified splitting. Experiments were conducted on an NVIDIA RTX 3080 Ti GPU (12 GB VRAM) running Ubuntu 20.04.

---

## Contact

- Qingchuan Zhang: zhangqingchuan@btbu.edu.cn
- Jian Li: lijian@nifdc.org.cn
