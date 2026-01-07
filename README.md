# WiDS-Midterm-Assignments : Python Data Analysis & Machine Learning

This repository contains all coding work completed up to the midterm exam.  
The assignments focus on **NumPy, Pandas, data visualization, exploratory data analysis (EDA), and basic machine learning models** using real-world datasets.


---

## Assignment 1: NumPy & Pandas Fundamentals

**File:** `assignment1_Q.1_numpy.py` & `assignment1_Q.2_videogames.py`

### Topics Covered
- NumPy array creation and manipulation
- Statistical operations
- Boolean masking
- Custom spiral traversal of a matrix
- Data analysis using Pandas
- Data visualization using Matplotlib

### Tasks Implemented
1. Created a 5×5 NumPy array with random integers
2. Extracted the middle element
3. Computed the mean of each row
4. Identified elements greater than the overall mean
5. Implemented a function to print elements in spiral order
6. Analyzed a video game sales dataset:
   - Computed global sales
   - Sorted games by total sales
   - Aggregated sales by genre
   - Visualized genre-wise sales
   - Filtered and analyzed *Grand Theft Auto* titles
   - Created a regional sales pie chart

## Assignment 2: Exploratory Data Analysis & Machine Learning

**File:** `assignment2_spotify_ml.py`

### Topics Covered
- Exploratory Data Analysis (EDA)
- Handling missing values
- Feature scaling (Standardization)
- Data visualization (Seaborn & Matplotlib)
- Supervised machine learning
- Model evaluation

### Tasks Implemented
1. Initial dataset inspection (shape, columns, info, statistics)
2. Identification of numerical and categorical features
3. Handling missing values in key audio features
4. Feature scaling using `StandardScaler`
5. Visualizations:
   - Histograms
   - Boxplots
   - Scatter plots
   - Correlation heatmap
6. Created a binary **mood** label using valence
7. Trained and evaluated:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree Classifier
8. Compared models using:
   - Classification reports
   - Confusion matrices

### Key Findings
- Logistic Regression provided the most balanced performance across precision, recall, and F1-score.
- KNN performed competitively but is sensitive to parameter selection.
- Decision Trees were interpretable but showed signs of overfitting at higher depths.

## Dataset
- `dataset.csv` contains the audio features dataset used for EDA and machine learning.
- Sales data in Assignment 1 is loaded directly from the given source.





