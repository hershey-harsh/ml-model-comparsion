# ML Model Comparison

This project compares the performance of two machine learning models—Logistic Regression and Decision Tree—on the Wine dataset from scikit-learn. It performs 5-fold stratified cross-validation, and conducts a paired t-test to evaluate statistical significance.

---

## Results - [https://docs.google.com/presentation/d/1H6vodnf2AEUPa9T9m_X9SOGliMaHBLPhyqALZwln_IY/edit?usp=sharing](https://docs.google.com/presentation/d/1H6vodnf2AEUPa9T9m_X9SOGliMaHBLPhyqALZwln_IY/edit?usp=sharing)

* **Mean Accuracy**:

  * Logistic Regression: \~0.97
  * Decision Tree: \~0.89
* **Statistical Test**:

  * t-statistic: \~X.XXX, p-value: \~X.XXX
  * 95% CI for difference: \[lower, upper]

> *Note: Exact values may vary slightly due to random splits.*

---

## How to run?

### Prerequisites

* Python 3.8 or higher
* pip

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/hershey-harsh/ml-model-comparsion
   cd ml-model-comparsion
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Experiment

To run the experiment and generate results:

```bash
python experiment.py
```

This will:

1. Load the Wine dataset.
2. Train and evaluate Logistic Regression and Decision Tree models using 5-fold stratified cross-validation.
3. Compute a paired t-test and 95% confidence interval on accuracy differences.
4. Save a boxplot of accuracy distributions to `output/accuracy_boxplot_wine.png`.
