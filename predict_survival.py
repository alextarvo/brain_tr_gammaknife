import argparse
from types import SimpleNamespace
import os
import logging
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import sklearn.model_selection as ms
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


def get_args():
    """
     Sets up command line arguments parsing
  """
    parser = argparse.ArgumentParser()
    parser.add_argument('--lesion_metrics_file',
                        help='Path to the clinical info file Brain-TR-GammaKnife-Clinical-Information.xlsx',
                        type=str)
    args = parser.parse_args()
    return args

def select_variables_rf(X, y, ntrees, max_tree_depth, visualize=False):
    model = RandomForestClassifier(
        n_estimators=ntrees,
        max_depth=max_tree_depth).fit(X, y)

    metrics_gini = model.feature_importances_
    
    predictor_columns = X.columns.tolist()
    assert (len(predictor_columns) == len(metrics_gini))
    features_importance_list = list(zip(metrics_gini, predictor_columns))
    features_importance_list = sorted(features_importance_list, key=lambda x: abs(x[0]), reverse=True)
    gini_coefficients_sorted, feature_names_sorted = zip(*features_importance_list)

    top_N_features = feature_names_sorted[0:TOP_N_METRICS]
    top_N_features_gini = gini_coefficients_sorted[0:TOP_N_METRICS]
    top_metrics = X[list(top_N_features)]
    if visualize:
        correlation_matrix_top_N = np.corrcoef(top_metrics, rowvar=False)
        correlation_matrix_top_N_plot = plt.figure(figsize=(12, 10))
        plt.imshow(correlation_matrix_top_N, cmap='seismic', vmin=-1, vmax=1)
        # plt.title(f"Cross-Correlation Matrix for top {self._top_metrics_to_pick} features")
        plt.title(f"Feature cross-correlation for survival prediction")
        plt.yticks(range(top_metrics.shape[1]), top_metrics.columns)
        plt.subplots_adjust(left=0.4, bottom=0.1, right=0.9, top=0.9)
        plt.show()

    return top_N_features, top_N_features_gini, top_metrics


def get_random_splits(X, y, n_splits):
    """
    Generates stratified train-test splits while ensuring at least `min_recurrent` recurrent patients in training.
    """
    splits = []
    skf = ms.StratifiedKFold(n_splits=n_splits)
    split_no = 1
    for train, test in skf.split(X, y):
        splits.append((X.iloc[train], y.iloc[train], X.iloc[test], y.iloc[test]))
        print(f'Split: {split_no}, train size: {len(train)}, test size: {len(test)}. Num. positives in train: {sum(y.iloc[train])}, in test: {sum(y.iloc[test])}')
        split_no += 1
    return splits

    # np.random.seed(random_state)
    #
    # for i in range(n_splits):
    #     while True:
    #         X_train, X_test, y_train, y_test = train_test_split(
    #             X, y, test_size=test_size, stratify=y, random_state=random_state + i
    #         )
    #
    #         # Ensure sufficient recurrent patients in the training set
    #         if y_train.sum() >= min_recurrent:
    #             splits.append((X_train, X_test, y_train, y_test))
    #             break  # Valid split found

    return splits

TOP_N_METRICS = 20
N_SPLITS = 10

if __name__ == "__main__":
    args = get_args()
    df_lesion_metrics = pd.read_csv(args.lesion_metrics_file, header=0, index_col=0)
    df_lesion_metrics['target'] = df_lesion_metrics['MRI_TYPE'].map({'recurrence': 1, 'stable': 0})
    df_lesion_metrics['gender_id'] = df_lesion_metrics['PATIENT_GENDER'].map({'Male': 1, 'Female': 0})

    X = df_lesion_metrics.drop(
        columns=['target', 'PT_ID', 'LESION_COURSE_NO', 'LESION_NO', 'PATIENT_DIAGNOSIS_PRIMARY',
                 'PATIENT_AGE', 'PATIENT_GENDER', 'DURATION_TO_IMAG', 'MRI_TYPE'])
    # This is an index column
    # X = X.iloc[:, 1:]
    y = df_lesion_metrics['target']
    # top_N_features, top_N_features_gini, top_metrics = select_variables_rf(
    #     X, y, ntrees=20, max_tree_depth=10)

    splits = get_random_splits(X, y, 10)
    model = AdaBoostClassifier(n_estimators=80)
    # model = RandomForestClassifier(n_estimators=80,
    #                                min_impurity_decrease=0.1,
    #                                class_weight={0:1, 1:5})  # Handles class imbalance
    for i, (X_train, y_train, X_test, y_test) in enumerate(splits):
        # Fit model to the current split
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Evaluate performance
        print(f"=== Split {i + 1} ===")
        print(classification_report(y_test, y_pred, target_names=["Stable", "Recurrent"]))
        print(f'Predicted: {y_pred}')
        print(f'Real:      {np.asarray(y_test)}')
        print("\n")
