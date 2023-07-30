# Mental Health in Tech Survey Analysis

## Project Description

This project aims to uncover patterns and trends related to mental health in the tech industry. It uses data from a survey where individuals working in the tech industry have provided answers about their remote work practices and mental health consequences they experience. The main goal of this project is to apply unsupervised machine learning techniques, specifically K-means clustering, to identify different groups within the data and then analyze the characteristics of these groups. 

## Project Structure

The project contains three main Python files: `clustering.py`, `visualization.py` and `main.py`.

- `clustering.py`: This file contains the function to perform K-means clustering. It also includes data preprocessing steps, such as scaling and PCA (Principal Component Analysis). After the clusters are identified, this file visualizes the clusters, the distribution of `remote_work` and `mental_health_consequence` within each cluster, and performs Chi-square tests to find relationships between these two variables.

- `visualization.py`: This file contains the function to visualize the distributions of `remote_work` and `mental_health_consequence` variables within each cluster.

- `main.py`: This is the main file that runs the whole analysis. It loads the data, calls the clustering function, and applies the Chi-square tests. 

## Requirements

Python 3.11.4 is required to run the code in this project. The following Python libraries are also required:
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
matplotlib==3.4.2
seaborn==0.11.1
scipy==1.7.0


As first step, you need to install these libraries using pip:

```
pip install -r requirements.txt
```

## Usage

To run the project:

1. Clone the repository to your local machine.
2. Navigate to the project directory and install the required libraries using the command `pip install -r requirements.txt`.
3. Run the `main.py` script to perform the clustering and visualizations.

## Data

The data used in this project comes from a tech industry survey, stored in a CSV file. The file contains multiple columns, but our analysis focuses on two: `remote_work` and `mental_health_consequence`.

