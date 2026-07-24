# Assignment 4 — Breast Cancer Classification using K-Nearest Neighbors (KNN)

**Name:** Adit Singh Rathore
**Registration Number:** 23BCE10238

## Objective

To build a K-Nearest Neighbors (KNN) classification model that predicts whether a breast tumor is **Malignant (M)** or **Benign (B)** using diagnostic measurements from the Breast Cancer Wisconsin dataset.

## Dataset Link

Breast Cancer Wisconsin (Diagnostic) Dataset (Kaggle):
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

> The dataset is **not** included in this repository as per the assignment instructions. The notebook uses `sklearn.datasets.load_breast_cancer()`, which contains the same data from the original UCI dataset. The feature names have been renamed to match the Kaggle dataset. If required, the Kaggle CSV can be downloaded from the above link and loaded using `pd.read_csv()`.

## Libraries Used

* **pandas** – Data loading and manipulation
* **numpy** – Numerical operations
* **scikit-learn** – Data preprocessing, model training, and evaluation
* **matplotlib** and **seaborn** – Confusion matrix visualization

## Methodology

1. **Data Understanding**

   * Loaded the dataset.
   * Viewed the first few records.
   * Examined the dataset using `.info()` and `.describe()`.
   * Identified the input features and target variable (`diagnosis`).

2. **Data Preprocessing**

   * Checked for missing values.
   * Removed the `id` column.
   * Converted the diagnosis labels into numeric values (Malignant = 1, Benign = 0).
   * Standardized all features using `StandardScaler`.
   * Split the dataset into 80% training data and 20% testing data.

3. **Model Development**

   * Trained a `KNeighborsClassifier` with **K = 5** using the training data.
   * Generated predictions on the test dataset.

4. **Model Evaluation**

   * Evaluated the model using Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix.

5. **Conclusion**

   * Summarized the model performance, importance of feature scaling, and limitations of KNN.

## Results

The model was evaluated on **114 test samples**.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.9561 |
| Precision | 0.9744 |
| Recall    | 0.9048 |
| F1-Score  | 0.9383 |

### Confusion Matrix

|                      | Predicted Benign | Predicted Malignant |
| -------------------- | :--------------: | :-----------------: |
| **Actual Benign**    |        71        |          1          |
| **Actual Malignant** |         4        |          38         |

### Observations

1. The KNN model performed well and correctly classified most benign and malignant tumors. The high accuracy, precision, recall, and F1-score show that the model makes reliable predictions.

2. Recall for the **Malignant** class is especially important. Predicting a malignant tumor as benign (false negative) can delay treatment, making it the most serious type of error. In this model, there were only **4 false negatives**.

3. Feature scaling played an important role in improving the model. Since KNN uses the distance between data points, features with larger values, such as `area_mean`, would have a much greater influence than smaller-valued features like `smoothness_mean` if the data were not scaled.

## Conclusion

This project used the K-Nearest Neighbors (KNN) algorithm to classify breast tumors as **malignant** or **benign** using the Breast Cancer Wisconsin (Diagnostic) dataset. After removing the ID column, converting the diagnosis labels into numbers, scaling all features, and splitting the data into training and testing sets, the KNN model with **K = 5** achieved **95.6% accuracy**, **97.4% precision**, **90.5% recall**, and **93.8% F1-score**.

Feature scaling was an important step because KNN makes predictions based on the distance between data points. Without scaling, features with larger values, such as `area_mean`, would have a greater influence on the distance calculation than features with smaller values, which could reduce the model's accuracy.

One limitation of KNN is that it stores all the training data and compares a new data point with every training example during prediction. As the dataset grows larger, this makes the model slower and requires more computation compared to models like Logistic Regression, which learn a fixed model during training.
