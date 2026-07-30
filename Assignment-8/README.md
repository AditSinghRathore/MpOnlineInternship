# Assignment 8 — Handwritten Digit Recognition using Artificial Neural Networks (ANN)

**Name:** Adit Singh Rathore
**Registration Number:** 23BCE10238

## Objective

A postal service organization wants to automate the recognition of handwritten digits on postal codes. This project develops an Artificial Neural Network (ANN) using TensorFlow/Keras to classify handwritten digits (0–9) from the MNIST dataset.

## Dataset Link

[MNIST Handwritten Digits Dataset — Kaggle (mnist-in-csv)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

> The dataset is **not included** in this repository. Download `mnist_train.csv` and `mnist_test.csv` from the Kaggle link above and place them in a `data/` folder before running the notebook/script.

## Libraries Used

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — train/test split, evaluation metrics
- `tensorflow` / `keras` — building and training the ANN

## Methodology

1. **Data Understanding** — Loaded the dataset with Pandas, inspected the first five records, identified input features (784 pixel columns) and the target variable (`label`), reviewed dataset dimensions/summary statistics, and visualized sample digit images.
2. **Data Preprocessing** — Checked for missing values (none found), separated features from the target, normalized pixel values from the 0–255 range to 0–1, split the data into 80% training / 20% testing using a stratified split, and one-hot encoded the target labels into 10-class categorical vectors.
3. **Model Development** — Built a Sequential ANN in Keras and trained it for 10 epochs using the Adam optimizer and categorical crossentropy loss.
4. **Model Evaluation** — Evaluated test accuracy, generated a confusion matrix and classification report, and plotted accuracy/loss curves across epochs.
5. **Conclusion** — Summarized key findings, the role of hidden layers, and a comparison between Deep Learning and traditional ML.

## Model Architecture

| Layer | Type | Units | Activation |
|---|---|---|---|
| Input | Dense input | 784 | — |
| Hidden Layer 1 | Dense | 128 | ReLU |
| Hidden Layer 2 | Dense | 64 | ReLU |
| Output Layer | Dense | 10 | Softmax |

**Compilation settings:**
- Optimizer: `Adam`
- Loss Function: `Categorical Crossentropy`
- Metric: `Accuracy`
- Epochs: `10` | Batch size: `128`

## Results

- **Test Accuracy:** 97.26%
- **Test Loss:** 0.1080

**Classification Report:**

```
              precision    recall  f1-score   support

           0     0.9949    0.9806    0.9877      1185
           1     0.9773    0.9881    0.9827      1348
           2     0.9513    0.9824    0.9666      1192
           3     0.9879    0.9299    0.9580      1226
           4     0.9785    0.9743    0.9764      1168
           5     0.9731    0.9677    0.9704      1084
           6     0.9889    0.9780    0.9834      1184
           7     0.9722    0.9777    0.9749      1253
           8     0.9339    0.9778    0.9553      1170
           9     0.9705    0.9681    0.9693      1190

    accuracy                         0.9726     12000
   macro avg     0.9728    0.9725    0.9725     12000
weighted avg     0.9730    0.9726    0.9726     12000
```

**Observations:**

1. **High overall accuracy with a simple architecture.** The ANN reaches ~97.3% test accuracy after just 10 epochs, showing that even a fully-connected network without convolutional layers performs well on MNIST, since the digits are centered, size-normalized, and relatively low-resolution (28×28).
2. **Training and validation curves track closely.** Accuracy and loss curves for training and validation stay close together across epochs, with validation loss flattening while training loss keeps dropping slightly — a sign of mild overfitting beginning to emerge, though not severe within 10 epochs.
3. **Confusable digit pairs.** The confusion matrix shows most misclassifications occur between visually similar digits, most commonly **4 and 9**, **3 and 5**, and **7 and 1** — digits that share loop, curve, or stroke similarities when handwritten.
4. **Class-wise performance is fairly uniform.** Per-class accuracy stays above 96% for nearly all digits, with structurally distinct digits like **0** and **1** classifying most reliably, while digits with more stroke variability (like **8** and **5**) show marginally more errors.

*(Full plots — sample digit grid, accuracy vs. epoch, loss vs. epoch, and confusion matrix heatmap — are available in `Assignment-8.ipynb`.)*

## Conclusion

This project developed a fully-connected Artificial Neural Network to classify MNIST handwritten digits, achieving strong test accuracy (~97.3%) after only 10 training epochs, demonstrating that ANNs can automate handwritten postal-code digit recognition effectively. The two hidden layers (128 and 64 neurons with ReLU activation) were essential for learning increasingly abstract, non-linear representations of pixel patterns — without them, the model would reduce to a linear classifier incapable of separating visually similar digits. A key advantage of Deep Learning over traditional Machine Learning here is automatic feature extraction: the ANN learns useful pixel-pattern features directly from raw data, whereas classical ML models typically need hand-engineered features. However, a notable limitation of this ANN is that it treats the image as a flat vector of pixels, discarding spatial structure; a Convolutional Neural Network (CNN) would likely outperform it by explicitly modeling local spatial patterns such as edges and curves.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
# Place mnist_train.csv and mnist_test.csv inside a data/ folder
jupyter notebook Assignment-8.ipynb
```
