# Salary Prediction using Polynomial Regression

## Objective

The objective of this assignment is to predict an employee's salary based on their position level. Since the relationship between position level and salary is not linear, a Polynomial Regression model is used to capture the curved relationship more accurately.

## Dataset Link

The Position_Salaries dataset was taken from Kaggle:

https://www.kaggle.com/datasets/akram24/position-salaries

*(As instructed, the dataset is not uploaded to GitHub. Only the dataset link is provided.)*

## Libraries Used

* pandas
* numpy
* matplotlib
* scikit-learn (sklearn)

## Methodology

1. Loaded the dataset using Pandas and explored it by viewing the first five rows, dataset information, and summary statistics.
2. Checked for missing values and found that the dataset contained no missing data. Selected **Level** as the input feature and **Salary** as the target variable.
3. Split the dataset into 80% training data and 20% testing data using `train_test_split` with `random_state=42`.
4. Transformed the input feature using `PolynomialFeatures` with degree 3 and trained a Linear Regression model on the transformed data.
5. Predicted salaries for the test data and evaluated the model using MAE, MSE, and R² Score. A scatter plot and the polynomial regression curve were also created for visualization.

## Results

* **MAE:** 70635.24
* **MSE:** 6263853282.86
* **R² Score:** 0.876

The model achieved an R² score of approximately **0.88**, indicating a good fit for the given dataset. However, since the dataset contains only 10 records, the evaluation results may vary slightly. The regression curve closely follows the original data points, especially at higher position levels where salaries increase rapidly.

## Conclusion

In this assignment, I used Polynomial Regression to predict an employee's salary based on their position level. The dataset contained only 10 records, but the relationship between position level and salary was clearly non-linear. Because of this, Linear Regression was not able to fit the data accurately.

Linear Regression fits a straight line to the data, while Polynomial Regression adds higher-degree terms (such as x² and x³) to create a curved model. This allows it to capture non-linear patterns more effectively.

In this dataset, salaries increase gradually at lower position levels but rise sharply at higher levels. Polynomial Regression is better suited for this type of relationship because it can model the rapid increase in salary more accurately than a simple linear model.

Overall, the Polynomial Regression model provides accurate predictions for this dataset. However, since the dataset is very small, its ability to generalize to new data may be limited.
