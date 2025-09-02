# 🌍 Land Cover Classification & Accuracy Analysis  

This repository provides an **interactive Gradio app** for classifying land cover types from **Sentinel-2 satellite imagery** and evaluating classification accuracy with standard remote sensing metrics.  

---

## 🚀 Features
- Upload **Sentinel-2 raster image (.tif)** and **label image (.tif)**  
- Classify land cover into 9 categories:  
  - Water  
  - Trees  
  - Grass  
  - Flooded Vegetation  
  - Crops  
  - Shrub & Scrub  
  - Built Area  
  - Bare Ground  
  - Snow & Ice  
- Adjustable parameters: **window size**, **sampling ratio**, **sampling method (Stratified / Random)**  
- Outputs include:  
  - **Exploratory Data Analysis (EDA)** plots & tables  
  - **Classification report** (precision, recall, F1-score)  
  - **Confusion matrix** (heatmap + styled table)  
  - **Overall accuracy metrics** (OA, Kappa, Tau, Z-statistic, confidence intervals)  
  - **Excel report** with confusion matrix & accuracy results  

---

## 🛠️ Installation

Clone the repository:  
```bash
git clone https://github.com/dayush3104/Land-Cover-Classification-and-Accuracy-Analysis.git
cd Land-Cover-Classification-and-Accuracy-Analysis
