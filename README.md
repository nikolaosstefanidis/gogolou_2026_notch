# 🗂️ My Project

![License](https://img.shields.io/badge/license-GNU_General_Public_License_v3.0-blue)
![PyPI version](https://img.shields.io/pypi/v/your-package)

## 🌟 Highlights

<table style="border: none; border-collapse: collapse;">
  <tr>
    <td valign="middle" style="border: none;">
      <ul>
        <li>Computational model of cell differentiation</li>
        <li>Approximate Bayesian Computation</li>
        <li>Jensenshannon analysis of posteriors</li>
      </ul>
    </td>
  </tr>
</table>

## ℹ️ Overview

The enteric nervous system (ENS) develops through a tightly regulated process in which bipotent progenitor cells differentiate into neurons and glia. Although the Notch signalling pathway is known to influence these transitions, the timing and rates of these changes in human cells remain poorly understood.
Using an in vitro human pluripotent stem cell differentiation platform, we generated longitudinal data on ENS development. We then built a computational model based on coupled ordinary differential equations to describe transitions rates between cellular states, and used Bayesian inference to quantify how Notch signalling affects these rates.
Our results show that Notch acts as a critical brake on differentiation. When Notch signalling is reduced, the balance of the system shifts, leading to premature ENS differentiation. This quantitative framework helps explain how altered developmental timing may contribute to neurological gut disorders such as Hirschsprung’s disease.

### ✍️ Authors

Nikolas Stefanidis (PhD student at the Maths & Stats school of the University of Sheffield, UK)  
Check out [my GitHub profile](https://github.com/nikolaosstefanidis)!  
Stanley Strawbridge (Group leader and Physics of Life fellow at the University of Sheffield, UK)  
Check out [Stan's GitHub profile](https://github.com/stanleystrawbridge)!
Alexander G Fletcher (Professor of Mathematical Biology and Head of Mathematical and Physical Sciences at the University of Sheffield, UK)  
Check out [Alex's GitHub profile](https://github.com/AlexFletcher)!

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A Google account with Google Drive (code is designed for Google Colab)
---

### Repository Structure
```
gogolou_2026_notch/
├── src/
│   ├── compile_data.py
│   ├── construct_model.py
│   ├── parameter_inference.py
│   ├── plotting.py
│   └── statistical_analysis.py
├── empirical_data/
├── results/
└── main.ipynb
```
### src folder  
Contains all functions necessary for the project

### Input data format

Each CSV in `empirical_data/` must contain the following columns:

| Column | Description |
|--------|-------------|
| `condition` | Experimental condition (`dmso` or `dapt`) |
| `day` | Timepoint (e.g. 9, 15, 22) |
| `bio_rep` | Biological replicate identifier |
| `tech_rep` | Technical replicate identifier |
| `total_cells` | Total cell count |
| `total_sox10` | SOX10-positive cell count - ONLY IN FIRST CSV |
| `total_s100b` | SOX10/S100B double-positive cell count - ONLY IN FIRST CSV |
| `total_hucd` | HuCD-positive cell count - ONLY IN SECOND CSV |

Two separate CSVs are expected — one for the SOX10/S100B staining and one for HuCD — loaded in that order.

---
### Running the analysis

Open and run `notebooks/main.ipynb` in order. The notebook is divided into clearly labelled sections:

| Step | What it does |
|------|-------------|
| Prepare folder | Download zip. When setting up the extract path remove '\gogolou_2026_notch-main' automatically added at the end. If not, you will get a 'gogolou_2026_notch-main' folder within a 'gogolou_2026_notch-main' folder. Either way it's import you use the folder immediately containing the package files. |
| Upload | Upload the correct 'gogolou_2026_notch-main' folder under My Drive (on Google Drive) |
| Setting up the main script | Open script.ipynb and follow the instructions on mounting your google drive to the google colab notebook |
| Set up paths | Follow instructions under User inputs to set up your paths |
| Hit Run all | This will run the entire script that generated data and figures for the manuscript |
| Load data | Reads all CSVs from `empirical_data/`, computes neural / progenitor / glial fractions |
| ABC – Untreated | Runs 10⁶-sample ABC inference against DMSO data across all 4 model variants |
| Filter posteriors | Accepts the top 0.2% of samples by distance `D` | 
| Plotting | Generates manuscript figures 5D-H |
| Statistics | Computes Jensen–Shannon divergence + permutation tests for comparisons shown in figures 5D-H |
| Statistics | Computes descriptive statistics for Untreated posteriors 
| ABC – Notch inhibited | Runs ABC against DAPT data, with `p_initial` fixed at the DMSO-inferred mean of model medians |
| Plotting | Generates manuscript figures 5J–P, S4) |
| Statistics | Computes Jensen–Shannon divergence + permutation tests for comparisons shown in figures 5J-P |
| Statistics | Computes descriptive statistics for Notch inhibited posteriors

All outputs (`.png`, `.pdf`, `.csv`) are saved automatically to `results/`.

---
