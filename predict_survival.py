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
from sklearn.feature_extraction.text import TfidfVectorizer

import sksurv
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

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
        # plt.show()
        print('1')

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

def plot_survival_curve(X, y):
    surv_data=sksurv.util.Surv.from_arrays(event=y,time=X['DURATION_TO_IMAG'])
    time, survival_prob, conf_int = kaplan_meier_estimator(
        surv_data['event'], surv_data['time'], conf_type='log-log'
    )
    plt.step(time, survival_prob, where="post")
    plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.ylim(0, 1)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.plot()

def predict_recurrence_survival(splits):
    c_scores = []
    rsf = RandomSurvivalForest(n_estimators=40, min_samples_leaf=1)
    for split_idx, (X_train, y_train, X_test, y_test) in enumerate(splits):
        train_surv_data = sksurv.util.Surv.from_arrays(event=y_train, time=X_train['DURATION_TO_IMAG'])
        test_surv_data = sksurv.util.Surv.from_arrays(event=y_test, time=X_test['DURATION_TO_IMAG'])
        X_train = X_train.drop(columns=['DURATION_TO_IMAG'])
        X_test = X_test.drop(columns=['DURATION_TO_IMAG'])

        # Fit model to the current split
        # Weigh "event" samples to 10
        sample_weight = np.where(y_train == 1, 10.0, 1.0)
        rsf.fit(X_train, train_surv_data, sample_weight=sample_weight)

        risk_scores = rsf.predict(X_test)

        #
        # Here, plot Kaplan-Meier scores for survival vs. non-survival
        #
        pred_survival_funcs = rsf.predict_survival_function(X_test)
        event_occurred_idx = np.where(y_test == 1)[0]  # Indices where event = 1
        censored_idx = np.where(y_test == 0)[0]  # Indices where event = 0
        # num_events = len(event_occurred_idx)  # Count of actual events
        # selected_censored_idx = np.random.choice(censored_idx, size=num_events, replace=False)  # Random selection
        event_curves = pred_survival_funcs[event_occurred_idx]
        # censored_curves = pred_survival_funcs[selected_censored_idx]

        time, survival_prob, conf_int = kaplan_meier_estimator(
            train_surv_data['event'], train_surv_data['time'], conf_type='log-log'
        )
        # time, survival_prob, conf_int = kaplan_meier_estimator(
        #     test_surv_data['event'][censored_idx], test_surv_data['time'][censored_idx], conf_type='log-log'
        # )
        plt.figure(figsize=(8, 6))
        for i, sf in enumerate(event_curves):
            plt.step(sf.x, sf(sf.x), where="post", linestyle="-", color="red", alpha=0.7,
                     label="Event" if i == 0 else "")
        plt.step(time, survival_prob, where="post")
        plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")

        # for i, sf in enumerate(censored_curves):
        #     plt.step(sf.x, sf(sf.x), where="post", linestyle="--", color="red", alpha=0.7,
        #              label="Censored" if i == 0 else "")

        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.title(f"Survival Function: Non-Survivors (Red) vs. average. Split: {split_idx}")
        plt.legend()
        plt.show()

        c_index = concordance_index_censored(test_surv_data["event"], test_surv_data["time"], risk_scores)[0]
        print(f'c-score for split {split_idx}: {c_index}')
        c_scores.append(c_index)
    print(f'Mean c-scores: {np.mean(c_scores)}')

def predict_recurrence_classification(splits):
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


if __name__ == "__main__":
    args = get_args()
    df_lesion_metrics = pd.read_csv(args.lesion_metrics_file, header=0, index_col=0)
    df_lesion_metrics['target'] = df_lesion_metrics['MRI_TYPE'].map({'recurrence': 1, 'stable': 0})
    df_lesion_metrics['gender_id'] = df_lesion_metrics['PATIENT_GENDER'].map({'Male': 1, 'Female': 0})

    # Apply TF-IDF encoding
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=['brain','met','mets','with', 'ca'])
    text_features = vectorizer.fit_transform(df_lesion_metrics["PATIENT_DIAGNOSIS_METS"])
    print(vectorizer.get_feature_names_out())
    text_features_df = pd.DataFrame(text_features.toarray(), columns=vectorizer.get_feature_names_out())
    df_lesion_metrics = pd.concat([df_lesion_metrics.drop(columns=["PATIENT_DIAGNOSIS_METS"]), text_features_df], axis=1)


    X = df_lesion_metrics.drop(
        columns=['target', 'PT_ID', 'LESION_NO', 'PATIENT_DIAGNOSIS_PRIMARY', #'PATIENT_DIAGNOSIS_METS',
                 'PATIENT_GENDER', 'MRI_TYPE'])
    # This is an index column
    # X = X.iloc[:, 1:]
    y = df_lesion_metrics['target']
    # top_N_features, top_N_features_gini, top_metrics = select_variables_rf(
    #     X, y, ntrees=20, max_tree_depth=10)

    plot_survival_curve(X, y)

    splits = get_random_splits(X, y, 10)
    predict_recurrence_survival(splits)
