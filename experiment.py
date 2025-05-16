import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import os

def main():
    os.makedirs('output', exist_ok=True)

    X, y = load_wine(return_X_y=True)
    print("Wine Dataset Features:")
    print(X)
    print("\nWine Dataset Labels:")
    print(y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=10000),
        'Decision Tree': DecisionTreeClassifier()
    }
    print("\nModels being compared:", list(models.keys()))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {
        name: cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        for name, model in models.items()
    }
    print("\nCross-validation scores:")
    for name, score in scores.items():
        print(f"{name}: {score}")

    diff = scores['Logistic Regression'] - scores['Decision Tree']
    t_stat, p_value = ttest_rel(
        scores['Logistic Regression'],
        scores['Decision Tree']
    )
    mean_diff = np.mean(diff)
    sem = np.std(diff, ddof=1) / np.sqrt(len(diff))
    ci_low = mean_diff - 1.96 * sem
    ci_high = mean_diff + 1.96 * sem

    print("\nStatistical Analysis:")
    print(f"Mean difference: {mean_diff}")
    print(f"Standard Error: {sem}")
    print(f"95% Confidence Interval: [{ci_low}, {ci_high}]")
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")

    plt.figure()
    plt.boxplot([scores[name] for name in models], labels=list(models.keys()))
    plt.ylabel('Accuracy')
    plt.title('5-Fold CV Accuracy Comparison (Wine Dataset)')
    plot_path = 'output/accuracy_boxplot_wine.png'
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
