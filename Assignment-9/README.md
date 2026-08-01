# Assignment 9: Image Classification using CNN (Cats vs Dogs)

## Objective
Build a Convolutional Neural Network (CNN) to classify pet images into Cats and Dogs, helping an animal welfare organization automate image classification.

## Dataset Link
[Cats vs Dogs Classification Dataset – Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

> Dataset is not included in this repository per submission guidelines. Download it from the Kaggle link above using a Kaggle API token.

## Libraries Used
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn (metrics)
- Pillow (PIL)

## Methodology
1. **Data Understanding** – Explored folder structure, class balance, sample images, and raw image dimensions.
2. **Data Preprocessing** – Removed corrupt image files, resized all images to 128×128, normalized pixel values to [0, 1], and split into 80% training / 20% testing using Keras `ImageDataGenerator`.
3. **Model Development** – Built and trained a CNN for 10 epochs using the Adam optimizer and binary crossentropy loss.
4. **Model Evaluation** – Evaluated using accuracy, precision, recall, F1-score, confusion matrix, and accuracy/loss curves across epochs.

## CNN Architecture
| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 128 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dense | 128 neurons, ReLU |
| Output | 1 neuron, Sigmoid |

**Optimizer:** Adam · **Loss:** Binary Crossentropy · **Metric:** Accuracy · **Epochs:** 10

## Results
- Test Accuracy, Precision, Recall, and F1-Score are reported in the notebook output (Task 4).
- Confusion matrix and Accuracy/Loss vs Epoch plots are included in the notebook.
- See `sample_images.png`, `confusion_matrix.png`, and `accuracy_loss_curves.png` for saved output visuals.

## Conclusion
The CNN successfully learned to distinguish cats from dogs by extracting hierarchical visual features through convolution and pooling layers. Convolution layers detect local patterns (edges, textures), while pooling layers reduce dimensionality and add translation robustness. Compared to a plain ANN, the CNN's weight-sharing via convolution kernels makes it far more parameter-efficient for image data. Its main limitation is a dependence on large labeled datasets and GPU compute for effective training. Full details are in the notebook's conclusion cell.

## How to Run (Local / VS Code)
1. Install dependencies: `pip install tensorflow scikit-learn matplotlib seaborn pillow pandas numpy kaggle`
2. Get a Kaggle API token: Kaggle account → Settings → **Create New Token** (downloads `kaggle.json`).
3. Place `kaggle.json` in the same folder as `Assignment-9.ipynb`.
4. Open the notebook and run all cells in order — the dataset downloads into a local `./data` folder automatically.
5. Update `DATA_DIR` / `BASE_DIR` in Task 1 if the extracted folder structure differs from `data/PetImages/Cat`, `data/PetImages/Dog`.
6. A GPU is optional but recommended — CPU-only training for 10 epochs on ~20k images will be noticeably slower.
