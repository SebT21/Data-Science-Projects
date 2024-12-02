# ETF Clustering: Identifying ETFs for Rotational Momentum Strategies

This repository contains the code and report for a project completed as part of the **APS1051 Financial Data Analysis and Visualization** course during my Master's in Engineering (MEng) at the **University of Toronto**. The project focuses on using clustering techniques to identify ETFs with similar performance characteristics for optimizing rotational momentum strategies.

## Project Objective

The goal of this project was to classify ETFs based on **returns** and **volatility** using **hierarchical clustering** and **k-means clustering**. This analysis aimed to discover ETFs with similar risk-return profiles to the original 21 sector and 7 government bond ETFs proposed in class.

## Features

- **Clustering Techniques**: 
  - Hierarchical clustering with dendrogram visualizations for grouping ETFs.
  - K-means clustering for comparative analysis and flexible partitioning.
- **Performance Evaluation**: 
  - Analyzed Sharpe Ratios and cumulative returns to identify optimal configurations.
- **Rotational Momentum Program**: Tested lookback and holding periods, and weight distributions for enhanced ETF selection.

## Contributions

- **Data Collection**: Gathered historical ETF data from Yahoo Finance, covering January 2016 to January 2020, including personal-interest ETFs for expanded analysis.
- **Clustering Implementation**: Conducted hierarchical and k-means clustering, analyzing cluster quality using silhouette scores.
- **Rotational Momentum Analysis**: Evaluated parameters for lookback periods, holding periods, and weight configurations, identifying the optimal setup.
- **Report Writing**: Authored the report, documenting the methodology, results, and insights for practical application.

## Results

- **Clustering Analysis**:
  - Hierarchical clustering achieved a silhouette score of 0.668, identifying closely related ETFs across various sectors and asset classes.
  - K-means clustering achieved a silhouette score of 0.643, highlighting the robustness of this method in complementing hierarchical clustering results.
- **Optimal Rotational Momentum Configuration**:
  - Lookback period: 25 days
  - Holding period: 2 weeks (ending on Fridays)
  - Weights: 0.4 (short-term returns), 0.2 (long-term returns), 0.4 (volatility)
  - Achieved a Sharpe Ratio of 1.342 with approximately fourfold returns over the four-year period.

## Repository Contents

- **`Code.py`**: Python scripts for data preprocessing, clustering, and rotational momentum analysis.
- **`Final Report.pdf`**: The final project report, detailing the methodologies, findings, and insights.
- **`Outputs and Results/`**: Visualizations of clustering results and cumulative returns for the best configuration.
- - **`Development Project Presentation.pptx`**: Slightly different presentation format of the results, with a closer snapshot of some of the clusters.

## Future Work

- Incorporate additional measures beyond returns and volatility for more nuanced clustering.
- Explore advanced clustering algorithms like DBSCAN for identifying non-linear patterns.
- Extend the analysis to include ETFs from newer market segments or alternative asset classes.
