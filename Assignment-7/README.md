# Assignment 7 — Customer Segmentation using K-Means Clustering and PCA

**Name:** Adit Singh Rathore
**Registration Number:** 23BCE10238

## Objective

To segment mall customers into distinct groups based on their annual income and
spending behavior using K-Means Clustering, and to visualize these clusters in
two dimensions using Principal Component Analysis (PCA). The resulting segments
are intended to support targeted marketing campaigns for the mall's management.

## Dataset Link

Mall Customer Segmentation Dataset (Kaggle):
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

> The dataset is **not** included in this repository per the assignment
> instructions. Download `Mall_Customers.csv` from the Kaggle link above and
> place it in the repository root before running the notebook.

## Libraries Used

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` — plotting
- `seaborn` — statistical visualization
- `scikit-learn` — `StandardScaler`, `LabelEncoder`, `KMeans`, `PCA`

## Methodology

1. **Data Understanding** — Loaded the dataset, inspected the first five
   records, identified numerical (`Age`, `Annual Income`, `Spending Score`)
   and categorical (`Genre`) features, and reviewed summary statistics.
2. **Data Preprocessing** — Checked for missing values, dropped the
   non-predictive `CustomerID` column, label-encoded `Genre`, and standardized
   all numerical features with `StandardScaler` so that features with larger
   numeric ranges (like income) don't dominate the distance calculations
   K-Means relies on.
3. **Model Development** — Used the Elbow Method (WCSS vs. K, for K = 1–10)
   to identify the optimal number of clusters, trained a `KMeans` model with
   the selected K, assigned a cluster label to every customer, and applied
   PCA to compress the standardized feature set into 2 principal components.
4. **Visualization and Evaluation** — Plotted the elbow curve, a scatter plot
   of clusters on Annual Income vs. Spending Score, and a PCA-based 2D
   scatter plot colored by cluster.

## Results

- The elbow curve identified **K = 5** as the optimal number of clusters.
- Five customer segments emerged, differing along income and spending
  behavior — for example, high-income/high-spending customers, high-income
  but low-spending customers, and low-income/high-spending customers.
- The 2 principal components captured a majority of the variance in the
  standardized feature set, and produced visibly well-separated clusters
  when plotted in 2D.

*(Exact cluster-mean values and the variance-explained percentage are printed
in the notebook output — these will differ slightly depending on the actual
Kaggle dataset values used.)*

## Conclusion

This project applied K-Means clustering to segment mall customers based on
their annual income and spending score, using the Elbow Method to identify
the optimal number of clusters. The resulting segments revealed distinct
customer profiles, each representing a different marketing opportunity, from
high-value customers worth retaining to high-income but disengaged customers
worth targeting with promotions. PCA was used to reduce the preprocessed
feature set to two dimensions, enabling clear visualization of cluster
separation that would otherwise be impossible to inspect directly. A key
limitation of K-Means is that it requires the number of clusters to be
specified in advance and assumes roughly spherical, similarly sized clusters.
PCA's main advantage in this workflow was reducing dimensionality while
preserving most of the variance in the data, making cluster quality
visually interpretable.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

1. Download `Mall_Customers.csv` from the Kaggle link above into the repo root.
2. Open `Assignment-7.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run all cells top to bottom.
