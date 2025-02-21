# How to run: 
Main file to run: exploration-and-preprocessing.ipynb

Everything has already been properly configured for you just need to run the cells again if you need to run code. 

Currently just use the exploration-and-preprocessing.ipynb as the main file for your data exploration etc. you can create another ipynb i could be wrong but as far as i remember the states dont transfer across notebooks so if you need anything within the exploration-and-preprocessing.ipynb u have to run it in that file for the cells

Alternative: 
You can write python scripts to call within the the ipynb files which might be easier for you to do but better to be done in jupyter

Suggestion: you can still create a new jupyter notebook for models but just need to reuse the same code for just importign dataset and headers the rest of the data exploration can ignore you will only need the items u find from it that's all 

# TO-DO LIST

## Classification: 
Target variable: LOS/NLOS 
Test variables: everything else(do data exploration to find out)


### Step 1: Data Exploration
Visualize features to understand their distribution, correlation, and relationship with the target variable.
Example tools:
Correlation Matrix: Check for highly correlated features.
Histograms/Boxplots: See the spread of each feature.
Scatter plots: Observe how features relate to LOS/NLOS.

### Step 2: Initial Feature Importance
Techniques to identify which features matter most:
Correlation Analysis: Strong correlation with the target is a good sign.
Domain Knowledge: Features known to be important (like CIR, FP_AMP, noise) should always be considered.

### Step 3: Feature Selection
Purpose: Narrow down to the most predictive features.
Techniques:
Statistical methods (e.g., Chi-squared, ANOVA).
Recursive Feature Elimination (RFE): Iteratively removes weak features.
Tree-based models (Random Forest): Automatically rank features by importance.
PCA: Reduces dimensionality, combining related features.

### Example
You start with all 15 features.
After correlation analysis and Random Forest feature importance, you find that:
CIR, FP_AMP1, STDEV_NOISE, and Measured range are very predictive.
Features like FRAME_LEN or RXPACC might not significantly impact prediction accuracy.
Decision: You use the top 5-8 features and ignore the less important ones.