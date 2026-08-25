# Environment-Aware 5G Signal Strength Prediction and Coverage Mapping Using Machine Learning

## Problem Statement

5G signal strength varies depending on geographical, environmental, and base-station characteristics such as terrain, building coverage, vegetation, distance from the base station, antenna configuration, and transmission power.

This project aims to develop a machine learning model to predict 5G signal strength (SS-RSRP) using these environmental and base-station factors. The predicted signal strength will then be used to generate a spatial coverage map to identify areas with potentially weak, moderate, or strong 5G coverage.

## Dataset

The project uses the **FishNet 30 × 30 m feature-engineered dataset** from the **Vienna 5G Macro-cell Signal Dataset**.

The dataset contains information related to environmental conditions, geographical characteristics, and base-station parameters that can influence 5G signal propagation.

## Prediction Target

**SS-RSRP (5G Received Signal Strength)**

SS-RSRP is used as the target variable for predicting the received 5G signal strength at different geographical locations.

## Key Features

The dataset includes features related to:

* Vegetation and environmental characteristics
* Building coverage and building height
* Distance from the base station
* Terrain and elevation
* Base-station transmission power
* Antenna configuration
* Frequency and bandwidth
* Base-station direction and angle
* Geographical and spatial characteristics

## Planned Machine Learning Workflow

The project will follow the following workflow:

1. Dataset exploration
2. Data preprocessing
3. Handling missing or inconsistent values
4. Feature selection
5. Exploratory data analysis
6. Machine learning model development
7. Model comparison
8. Model evaluation using suitable regression metrics
9. Selection of the best-performing model
10. Signal-strength prediction
11. Spatial 5G coverage mapping

## Machine Learning Models

Different regression models will be explored and compared for SS-RSRP prediction, such as:

* Linear Regression
* Random Forest Regression
* Gradient Boosting Regression

The best-performing model will be selected based on its prediction performance.

## Expected Output

The project aims to produce:

* A trained machine learning model for SS-RSRP prediction
* Predicted 5G signal-strength values
* Model performance comparison
* Visualization of prediction results
* A spatial 5G signal-strength coverage map
* Identification of areas with potentially weak and strong signal coverage

## Project Status

**Current Stage:** Initial project setup and dataset analysis.

Completed or initiated:

* Project structure setup
* Dataset identification
* Initial dataset exploration
* Target variable identification

Planned next steps:

* Data preprocessing
* Feature selection
* Exploratory data analysis
* Model training and comparison
* Model evaluation
* Coverage-map generation

## Project Goal

The overall goal is to demonstrate how machine learning can be used with environmental and base-station information to predict 5G signal strength and support data-driven analysis of wireless network coverage.
