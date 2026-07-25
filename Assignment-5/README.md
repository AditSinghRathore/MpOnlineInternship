# Assignment 5 - Employee Attrition Prediction

**Name:** Adit Singh Rathore
**Registration Number:** 23BCE10238

## Objective

The goal of this assignment was to predict whether an employee is likely to leave the company (this is called "Attrition") using their personal and work-related details, like age, salary, job satisfaction, and overtime. I built two models - a **Decision Tree** and a **Random Forest** - and compared how well each one did.

## Dataset Link

IBM HR Analytics Employee Attrition & Performance Dataset (Kaggle):
https://www.kaggle.com/datasets/pavansubhasht/ibm-hranalytics-attrition-dataset

*(Following the assignment instructions, the dataset itself is not uploaded to this repo - please download it from the Kaggle link above and place it inside the `data/` folder as `WA_Fn-UseC_-HR-Employee-Attrition.csv` before running the notebook.)*

## Libraries Used

- `pandas` - loading and cleaning the data
- `numpy` - numerical operations
- `matplotlib` and `seaborn` - plots and charts
- `scikit-learn` - splitting the data, building the models, and evaluating them

## Methodology

1. **Data Understanding** - loaded the dataset, looked at the first few rows, checked which columns are numbers vs categories, and found the target column (`Attrition`).
2. **Data Preprocessing** - checked for missing values (there were none), removed columns that don't help (like ID numbers or columns where every row had the same value), converted text columns into numbers using one-hot encoding, and split the data into 80% training / 20% testing.
3. **Model Development** - trained a Decision Tree Classifier and a Random Forest Classifier (100 trees) on the same training data.
4. **Model Evaluation** - compared both models using Accuracy, Precision, Recall, and F1-Score, and plotted confusion matrices for both plus a feature importance chart for the Random Forest.

## Results

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Decision Tree  | 0.62     | 0.29      | 0.29   | 0.29     |
| Random Forest  | 0.73     | 0.50      | 0.05   | 0.09     |

*(Exact numbers may vary slightly depending on the dataset used - the notebook re-calculates these automatically.)*

## Model Comparison

Random Forest scored a higher accuracy, but the Decision Tree actually caught more of the employees who really did leave (higher recall). This happened because there are way fewer "left the company" examples than "stayed" examples in the data, so Random Forest leaned towards just predicting "stayed" most of the time - which pumps up accuracy but hurts recall. This showed me that accuracy by itself doesn't always tell the full story, especially when the data is imbalanced.

## Conclusion

Random Forest got the higher accuracy in this assignment, but it wasn't clearly "better" overall since the Decision Tree caught more actual leavers. Random Forest usually does better than a single Decision Tree because it combines many trees instead of relying on just one, which normally makes it more stable and less likely to overfit. A limitation of Decision Trees is that a single tree can overfit and memorize small quirks in the training data. A limitation of Random Forest is that it's harder to interpret (since it's really 100 trees combined) and it can lean towards the majority class when the data is imbalanced, like it did here.

## Bonus Challenge

I tried changing `max_depth` on the Decision Tree (values: 3, 5, 7, and no limit) to see how it affects performance. Limiting the depth changed both accuracy and F1-Score compared to letting the tree grow fully - a shallower tree is simpler and less likely to overfit, but too shallow can miss important patterns. Full details and numbers are in the notebook.

## Files in this repo

- `Assignment-5.ipynb` - the full notebook with all code, outputs, and plots
- `README.md` - this file
- `images/` - saved copies of the charts (metric comparison, confusion matrices, feature importance)
- `data/` - folder for the dataset (not uploaded here, see Dataset Link above)
