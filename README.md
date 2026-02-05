# Predicting Spotify Track Popularity via Audio Features

This project investigates the relationship between a song's intrinsic audio attributes and its overall popularity on Spotify. By applying machine learning techniques to a comprehensive dataset of track characteristics, we aim to understand whether the technical DNA of a song can reliably predict its commercial success.


## Project Overview

Music popularity is often attributed to external factors like marketing budgets, artist fame, or social media trends. This study shifts the focus inward, examining technical features such as danceability, energy, loudness, and acousticness to determine their predictive power. We analyze how these predictors function across various musical genres and evaluate the effectiveness of different statistical learning models in categorizing tracks into popularity tiers.


## Research Questions

Our analysis is guided by three primary inquiries:

* How accurately can machine learning models predict a song’s popularity tier based solely on audio attributes?
* Which specific audio features serve as the strongest indicators of a track's potential popularity?
* Do the key predictors of success remain consistent across different musical genres, or are they genre-dependent?


## Methodology

### Data Preprocessing
The raw dataset required extensive refinement to ensure model stability and prevent bias:

* Deduplication: Removed identical tracks occurring across different album versions or singles.
* Outlier Removal: Filtered extreme values in tempo and duration to focus on standard musical tracks.
* Feature Engineering: Implemented interaction terms (e.g., loudness × genre) to capture how certain attributes behave differently depending on the musical context.
* Normalization: Applied PowerTransformer (Yeo-Johnson) for skewed distributions and MinMax scaling for bounded audio features.
* Balancing: Addressed the high volume of low-popularity tracks by undersampling and organizing popularity into 20-point bins (e.g., 0–20, 20–40, 40–60, 60–80, 80–100).

### Predictive Modeling
We compared two distinct approaches to capture both linear and nonlinear relationships:

* Logistic Regression: Served as a baseline linear model to assess straightforward correlations.
* Support Vector Machine (SVM): Utilized with a Radial Basis Function (RBF) kernel to identify complex, nonlinear boundaries in the high-dimensional feature space.


## Key Findings

### Model Performance
The SVM model consistently outperformed Logistic Regression across all key metrics, suggesting that the relationship between audio features and popularity is inherently nonlinear. For instance, attributes like loudness or energy may only correlate with success up to a certain threshold.

* Accuracy: SVM achieved approximately 61.7% compared to 56.6% for Logistic Regression.
* Macro-F1: SVM reached 0.515, highlighting its superior ability to handle the underlying data structure.

### The Popularity Ceiling
While audio features provide moderate predictive power for tracks in the 0–60 popularity range, they fail significantly when predicting superstar status (tiers 80–100). Our models achieved an F1-score of only 0.18 for the highest tier, confirming that extreme viral success depends heavily on non-audio factors like playlist placement, artist reach, and marketing influence.

## Conclusion

Audio attributes are a useful starting point for understanding track success and can reliably categorize songs into broad popularity groups. However, technical features alone do not capture the "lightning in a bottle" quality of top-tier hits. To achieve higher accuracy for the most popular songs, future models would need to integrate contextual metadata such as social media engagement and historical artist performance.

## Repository Structure

* Data Cleaning.ipynb: Initial processing, deduplication, and balancing.
* EDA.ipynb: Exploratory analysis and visualization of feature correlations.
* DataPreproc+models.ipynb: Feature scaling and model training (Logistic Regression vs. SVM).
* ModelEval+Interpretation.ipynb: Final evaluation metrics and research conclusions.
* cleaned_dataset_EDA.csv: The processed dataset used for analysis.


## Requirements

To run the notebooks in this repository, you will need:

* Python 3.x
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
