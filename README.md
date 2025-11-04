# timeseries-anomaly-s6
“Selective State Space (S6/Mamba) + STL detrending for robust time-series anomaly detection (NASA, SMD, SWaT).”
# Time-Series Anomaly Detection with Selective State Space Model (S6/Mamba) and STL Detrending

This repository reproduces and extends the official implementation of  
**“Joint Selective State Space Model and Detrending for Robust Time Series Anomaly Detection”**  
by *Chen et al., IEEE Signal Processing Letters, 2024*.

The project was carried out as part of the seminar  
**“Advanced Machine Learning for Anomaly Detection”** at  
**Friedrich-Alexander-Universität Erlangen–Nürnberg (FAU)**.

---

## Overview
Time-series anomaly detection (TSAD) aims to identify irregular behaviors in sequential sensor or industrial data.  
This work builds on the **Selective State Space (S6 / Mamba)** model with a multi-stage detrending mechanism.  
The focus of this project is to integrate and evaluate a new **Seasonal-Trend Decomposition using LOESS (STL)** filter  
in place of the original Hodrick–Prescott (HP) filter to improve robustness for **non-stationary** time-series data.

---

## Data and Scope
**Datasets Used**
- **NASA (SMAP & MSL)** — satellite sensor readings  
- **SMD (Server Machine Dataset)** — server log metrics  
- **SWaT (Secure Water Treatment)** — industrial control system data  
- (Optional) **Yahoo Webscope A1** — labeled synthetic and real anomalies  

Each dataset contains multivariate time-series data with annotated anomalies,  
commonly used as TSAD benchmarks.
---



## Implementation

### Baseline
The baseline follows the original S6 model combined with the HP trend filter for detrending.

### Extension
In this project:
- Implemented an **STLFilter** class using `statsmodels` and integrated it into the training pipeline.  
- Added a command-line flag `--detrend_method` to switch between **HP** and **STL** filters.  
- Conducted comparative experiments under identical configurations (window, batch size, optimizer).  
- Results were logged and saved for each dataset subset (e.g., `NASA_result.csv`, `SMD_result.csv`).

### Tools and Frameworks
- **Python 3.10**  
- **PyTorch** — model training and evaluation  
- **Statsmodels** — STL decomposition  
- **NumPy, Pandas, Matplotlib** — data processing and visualization  

---
---
---

---

##  Results — NASA Dataset

Below are comparison plots of Precision (P_af), Recall (R_af), and F1 (F1_af)  
between the baseline **HP filter** and the proposed **STL filter** on NASA subsets (A-4, C-2, T-1).

###  Recall Comparison
![NASA Recall Comparison](plots/NASA%20Recall.png)

###  F1 Score Comparison
![NASA F1 _af Comparison](plots/NASA%20F1.png)

###  Precision Comparison
![NASA P_af Comparison](plots/NASA%20Precision%20.png)

---

**Observations:**
- Both HP and STL achieve high recall (~0.98–1.00) across all subsets.  
- STL slightly improves F1 on subset T-1 and provides smoother trend–seasonality separation.  
- HP performs marginally better in precision on A-4, but overall robustness favors STL.


## Data Preparation and Model Run

Download the benchmark datasets and extract them into a local folder:

```bash
unzip -d ./datasets datasets.zip

---
---
## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/tsad-s6-stl.git
cd tsad-s6-stl
pip install -r requirements.txt

---
---

## Baseline (HP filter)

Run the original implementation using the Hodrick–Prescott (HP) filter:

```bash
python run_experiment.py --dataset NASA --detrend_method HP --tag hp

---
---

##Proposed Method (STL filter)

Run the modified implementation using the Seasonal-Trend Decomposition (STL) filter:

```bash
python run_experiment.py --dataset NASA --detrend_method STL --tag stl

---
---

## Results (Precision, Recall, and F1 scores) will be automatically saved in the ./results/ directory as CSV files.
Example:

results/
├─ NASA_result.csv
└─ SMD_result.csv


---
---

