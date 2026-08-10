# US Drought Analysis with PySpark

Large-scale analysis of drought conditions across the United States using **Apache Spark**, combining data processing, exploratory analysis, geospatial visualization, and unsupervised machine learning.

## Overview

This project analyzes US drought data from **2001–2021** to identify temporal and geographic drought patterns and group states according to similar drought profiles.

### What was done

* **Data Processing**: Data ingestion, validation, cleaning, transformation and aggregation using PySpark.
* **Exploratory Analysis**: Analysis of drought severity, affected area and population across states and over time.
* **Severity Index**: Creation of a weighted drought severity index based on drought level and affected area.
* **Geospatial Analysis**: Interactive maps showing drought severity and state-level patterns.
* **Machine Learning**: K-Means clustering to group states according to their drought profiles.
* **Model Evaluation**: Elbow Method, Silhouette Score and clustering stability analysis.
* **Feature Analysis**: Comparison of clustering results with and without the D0 drought level.

## Tech Stack

**Python · PySpark · Apache Spark · Spark MLlib · Pandas · NumPy · Matplotlib · Seaborn · Plotly**

## Machine Learning

States are represented using the percentage of affected area at each drought level:

```text
D0 · D1 · D2 · D3 · D4
```

The features are standardized and clustered using **K-Means**. Different values of `K` and random seeds are evaluated using the **Silhouette Score** and **WCSS**.

## Project Structure

```text
US_Droughts/
├── PySpark_PL04.py
├── Fase02_PL04.pdf
└── README.md
```

> The datasets are not included in the repository. Input paths must be configured in the script.

## Academic Context

Developed as part of a **Big Data Processing** project at **ISCTE – University Institute of Lisbon**.
Grade: 17/20
