# Hand Gesture Recognition using CNN

A **Convolutional Neural Network (CNN)** that classifies 10 different hand gestures from near-infrared images, enabling intuitive gesture-based human-computer interaction and control systems. Built on the Kaggle
["LeapGestRecog"](https://www.kaggle.com/gti-upm/leapgestrecog) dataset.

## Overview

| | |
|---|---|
| **Task** | Multi-class image classification (10 gesture classes) |
| **Model** | CNN — Conv2D + BatchNorm + SpatialDropout + GlobalAveragePooling |
| **Dataset** | LeapGestRecog — ~20,000 images, 10 subjects × 10 gestures |
| **Framework** | TensorFlow / Keras |

## Gesture Classes

```
01_palm, 02_l, 03_fist, 04_fist_moved, 05_thumb,
06_index, 07_ok, 08_palm_moved, 09_c, 10_down
```

## The Key Design Decision: Subject-Based Evaluation

A naive **random split** across all ~20,000 images lets near-identical consecutive video frames of the *same person* land in both the training and test sets. This inflates test accuracy to look almost perfect (~99.9%), but it doesn't reflect how the model would perform for a genuinely new user.

This project instead uses a **subject-based split** (`GroupShuffleSplit` / `GroupKFold`) — entire people are held out for testing, never split across train/test. This is the evaluation that actually matters for a real gesture-control system: *"does this work for someone the model has never seen before?"*

A side-by-side comparison model trained on the naive random split is included specifically to **demonstrate** the accuracy inflation caused by frame leakage, rather than just asserting it.

## Project Structure

```
├── Hand_Gesture_Recognition_CNN.ipynb   # Full notebook: loading, training, evaluation
└── README.md
```

(Dataset is downloaded automatically via `kagglehub` — no manual file setup needed.)

## Workflow

1. **Download dataset** — via `kagglehub.dataset_download`.
2. **Build label lookup** — dynamically from folder structure (no hardcoded class names).
3. **Load images** — resize to 64×64, tracking each image's **subject ID** alongside its gesture label.
4. **Preprocess** — normalize pixels to [0,1], one-hot encode the 10 gesture labels.
5. **Subject-based split** — `GroupShuffleSplit`, with an explicit check confirming zero subject overlap between train and test.
6. **Build CNN** — 3 Conv blocks (32→64→128 filters) with BatchNorm, MaxPooling, SpatialDropout, then GlobalAveragePooling → Dense(128) → Dropout → Dense(10, softmax).
7. **Data augmentation** — geometric transforms (rotation, shift, zoom) applied during training via `ImageDataGenerator.flow(...)`.
8. **Train** — Adam optimizer, EarlyStopping on validation loss.
9. **Evaluate** — accuracy, classification report, confusion matrix — on held-out **unseen subjects**.
10. **Comparison model** — same architecture trained on a naive random split, to show the leakage effect directly.
11. **Cross-validation** — `GroupKFold` (subject-aware) to confirm the honest accuracy is consistent across different held-out people.
12. **Visualize predictions** — sample test images with actual vs predicted labels.

## Results

> Fill in with your actual run output from Colab:

| Evaluation | Accuracy |
|---|---|
| Subject-based split (honest, unseen people) | _e.g. 0.XX_ |
| Random split (frame leakage likely) | _e.g. 0.XX_ |
| GroupKFold mean accuracy (subject-based) | _e.g. 0.XX ± 0.XX_ |

## Known Issue & Fix: Brightness Augmentation

`ImageDataGenerator`'s `brightness_range` rescales images to a 0–255 range internally (via a PIL round-trip), which conflicts with images already normalized to [0,1] before augmentation. This mismatch can cause the model to collapse to predicting a single class. **Fix:** brightness augmentation was removed, keeping only geometric augmentations (rotation, shift, zoom), which don't have this scaling conflict.

## Limitations

- Even the subject-based split only holds out 2 of 10 subjects for testing — a small number of subjects, so results may vary based on which subjects land in the test set.
- Trains 3+ full CNN models in total (main model, comparison model, cross-validation folds), so total runtime is significant — expect it to take a while, even on a Colab GPU.

## How to Run

1. Open `Hand_Gesture_Recognition_CNN.ipynb` in Google Colab.
2. Enable a GPU runtime (**Runtime → Change runtime type → GPU**).
3. **Runtime → Run all.** The dataset downloads automatically via `kagglehub` — no manual setup needed.
4. If your runtime disconnects partway through, use **Runtime → Run all** again rather than resuming from a single cell, since all variables reset.

## Requirements

```
opencv-python
numpy
matplotlib
seaborn
scikit-learn
tensorflow
kagglehub
```

Colab has all of these pre-installed except `kagglehub`, which the first cell installs/imports automatically.
