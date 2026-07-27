# Assignment 6 - Weather Condition Classification using SVM

**Name:** Adit Singh Rathore
**Registration Number:** 23BCE10238
**Assignment Number:** Assignment-6

## Objective

The goal of this assignment is to classify weather conditions as either **Cool** or **Warm** using a Support Vector Machine (SVM) classifier, based on real weather data pulled from the Open-Meteo API. This was my first time building an end-to-end ML project that fetches live data from an API instead of using a ready-made CSV file, so a lot of this was new to me.

## API Documentation Link

- Open-Meteo Forecast API: https://open-meteo.com/
- Endpoint used: `https://api.open-meteo.com/v1/forecast`
- No API key is required, which made it a lot easier to get started.

## Libraries Used

- `requests` - for calling the Open-Meteo API
- `pandas` - for storing and manipulating the data in a table (DataFrame)
- `numpy` - for numeric operations (like creating the target column)
- `matplotlib` and `seaborn` - for plotting graphs (class counts, confusion matrix)
- `scikit-learn` - for:
  - `train_test_split` (splitting data)
  - `StandardScaler`, `LabelEncoder` (preprocessing)
  - `SVC` (the SVM model itself)
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report` (evaluation)

## Methodology

1. **Data Collection:** Called the Open-Meteo `/v1/forecast` endpoint for New Delhi (latitude 28.6139, longitude 77.2090) requesting hourly `temperature_2m`, `relative_humidity_2m`, `surface_pressure`, and `wind_speed_10m` for 7 days. This gives 168 hourly rows of data. Converted the JSON response into a Pandas DataFrame.
2. **Target Variable:** Created a new column `Weather_Class` using a simple rule: temperature ≥ 25°C = "Warm", otherwise "Cool".
3. **Preprocessing:** Checked for missing values (there were none), dropped the `time` column since it isn't useful for prediction, encoded the target labels into numbers using `LabelEncoder`, split the data 80/20 into train/test sets, and standardized the 4 input features using `StandardScaler`.
4. **Model:** Trained a Support Vector Machine classifier (`SVC`) with an **RBF kernel** on the scaled training data.
5. **Evaluation:** Measured Accuracy, Precision, Recall, and F1-Score on the test set, and plotted a Confusion Matrix to see where the model made mistakes.

## Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.971 |
| Precision | 1.000 |
| Recall    | 0.962 |
| F1-Score  | 0.980 |

**Confusion Matrix:**

|              | Predicted Cool | Predicted Warm |
|--------------|-----------------|-----------------|
| **Actual Cool** | 8 | 0 |
| **Actual Warm** | 1 | 25 |

### Observations
1. The model achieved a high accuracy (~97%), meaning it could correctly tell apart Warm and Cool hours almost all the time using temperature, humidity, pressure, and wind speed.
2. Most errors (if any) happened around the "Cool" class boundary, likely because there were fewer Cool examples in the data (July in Delhi is mostly hot), giving the model less to learn from for that class.
3. Precision was very high (1.00), meaning whenever the model predicted "Warm" it was almost always correct - which makes sense since temperature (the exact feature the label is derived from) is one of the direct inputs to the model.

## Conclusion

This assignment helped me build my first SVM classification model using real-time weather data from the Open-Meteo API. I classified hourly weather readings from Delhi as either "Warm" or "Cool" based on temperature, and the SVM model with an RBF kernel achieved a high accuracy on the test data, showing that features like humidity, pressure, and wind speed do carry useful information related to temperature.

Feature scaling turned out to be very important for SVM. Since my features had very different ranges (temperature in tens, pressure in hundreds), the model could have been biased towards larger-valued features without `StandardScaler`. Scaling makes sure every feature contributes fairly to the model's decision.

One advantage of SVM is that it works well even with a relatively small number of samples, like the 168 hourly readings used here. One limitation is that SVM can get slower and harder to tune on very large datasets, and choosing the right kernel and parameters (like C and gamma) isn't always straightforward for a beginner.

## Files in this repository

- `Assignment-6.ipynb` - the full notebook with code, explanations, and outputs
- `open_meteo_delhi_raw.csv` - a saved backup copy of the weather data, used automatically as a fallback if the live API request fails (e.g. no internet or the request gets blocked)
- `README.md` - this file

## Note on Data Source

The notebook calls the **live Open-Meteo API** directly - no key needed. If for any reason the API can't be reached (blocked network, no internet, temporary API downtime), the notebook automatically falls back to the saved `open_meteo_delhi_raw.csv` file so it can still be run and graded end-to-end.
