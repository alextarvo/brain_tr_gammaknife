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
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

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
    parser.add_argument("--pca", action="store_true", help="Do PCA on numeric features")
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
    for train, test in skf.split(X, y['target']):
        splits.append((X.iloc[train], y.iloc[train], X.iloc[test], y.iloc[test]))
        no_pos_train = y.iloc[train]['target']
        no_pos_test = y.iloc[test]['target']
        print(
            f'Split: {split_no}, train size: {len(train)}, test size: {len(test)}. Num. positives in train: {sum(no_pos_train)}, in test: {sum(no_pos_test)}')
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


def plot_survival_curve(y):
    surv_data = sksurv.util.Surv.from_arrays(event=y['target'], time=y['DURATION_TO_IMAG'])
    time, survival_prob, conf_int = kaplan_meier_estimator(
        surv_data['event'], surv_data['time'], conf_type='log-log'
    )
    plt.step(time, survival_prob, where="post")
    plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.ylim(0, 1)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.grid()
    plt.plot()


def predict_recurrence_survival(splits):
    c_scores = []

    def average_survival_curves(surv_funcs):
        # Define a common time grid (using the union of all time points)
        time_grid = np.linspace(0, max(max(sf.x) for sf in surv_funcs), 100)  # 100 time points
        # Interpolate all survival curves on this grid
        surv_probs = np.array([np.interp(time_grid, sf.x, sf.y) for sf in surv_funcs])
        # Compute mean survival probability at each time point
        mean_surv_prob = np.mean(surv_probs, axis=0)
        # Compute confidence interval (e.g., 95% CI)
        lower_bound = np.percentile(surv_probs, 2.5, axis=0)  # 2.5th percentile
        upper_bound = np.percentile(surv_probs, 97.5, axis=0)  # 97.5th percentile
        return time_grid, mean_surv_prob, lower_bound, upper_bound

    def plot_averaged_survival_curves(all_censored_curves, all_event_curves):
        time_censored, mean_surv_prob_censored, lower_bound_censored, upper_bound_censored = average_survival_curves(
            all_censored_curves)
        time_event, mean_surv_prob_event, lower_bound_event, upper_bound_event = average_survival_curves(
            all_event_curves)

        plt.figure(figsize=(8, 6))
        # for i, sf in enumerate(event_curves):
        #     plt.step(sf.x, sf(sf.x), where="post", linestyle="-", color="red", alpha=0.7,
        #              label="Event" if i == 0 else "")
        # plt.step(time, survival_prob, where="post")
        # plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
        plt.step(time_event, mean_surv_prob_event, where="post", color="red", )
        plt.fill_between(time_censored, lower_bound_event, upper_bound_event, color="red", alpha=0.2,
                         label="True events")
        plt.step(time_censored, mean_surv_prob_censored, where="post", color="green", )
        plt.fill_between(time_censored, lower_bound_censored, upper_bound_censored, color="green", alpha=0.2,
                         label="True censored")

        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.ylim(0, 1)
        plt.title(f"Survival Function: Non-Survivors (Red) vs. censored (Green)")
        plt.legend()
        plt.show()

    all_event_curves = []
    all_censored_curves = []
    all_c_scores = []

    for split_idx, (X_train, y_train, X_test, y_test) in enumerate(splits):
        train_surv_data = sksurv.util.Surv.from_arrays(event=y_train['target'], time=y_train['DURATION_TO_IMAG'])
        test_surv_data = sksurv.util.Surv.from_arrays(event=y_test['target'], time=y_test['DURATION_TO_IMAG'])

        # Best for PyRadiomics + PCA:
        # rsf = RandomSurvivalForest(n_estimators=200,
        #                            max_depth=10,
        #                            max_features="sqrt",
        #                            bootstrap=True)

        # Best for fine-tuned model
        # rsf = RandomSurvivalForest(n_estimators=500,
        #                            min_samples_split=max(10, int(0.05 * len(y_train))),
        #                            min_samples_leaf=max(5, int(0.02 * len(y_train))),
        #                            max_depth=30,
        #                            max_features="sqrt",
        #                            bootstrap=True)

        # rsf = RandomSurvivalForest(n_estimators=40, min_samples_leaf=1)
        # rsf = RandomSurvivalForest(n_estimators=40,
        rsf = RandomSurvivalForest(n_estimators=800,
                                   min_samples_split=max(10, int(0.05 * len(y_train))),
                                   min_samples_leaf=max(5, int(0.02 * len(y_train))),
                                   max_depth=30,
                                   max_features="sqrt",
                                   bootstrap=True)

        # Fit model to the current split
        # Weigh "event" samples to 10
        # sample_weight = np.where(y_train == 1, 10.0, 1.0)
        sample_weight = np.where(y_train['target'] == 1, 8.0, 1.0)
        rsf.fit(X_train, train_surv_data, sample_weight=sample_weight)

        # risk_scores = rsf.predict(X_test)

        #
        # Here, plot Kaplan-Meier scores for survival vs. non-survival
        #
        # This is a "baseline" curve from tthe test set - all datapoints
        time, survival_prob, conf_int = kaplan_meier_estimator(
            train_surv_data['event'], train_surv_data['time'], conf_type='log-log'
        )
        # Get predicted survival functions for the "event" cases
        pred_survival_funcs = rsf.predict_survival_function(X_test)
        event_occurred_idx = np.where(y_test == 1)[0]  # Indices where event = 1
        event_curves = pred_survival_funcs[event_occurred_idx]
        all_event_curves.extend(event_curves)
        # And for censored cases
        censored_idx = np.where(y_test == 0)[0]  # Indices where event = 0
        censored_curves = pred_survival_funcs[censored_idx]
        all_censored_curves.extend(censored_curves)
        #
        # plt.figure(figsize=(8, 6))
        # for sf in event_curves:
        #     plt.step(sf.x, sf(sf.x), where="post", linestyle="-", color="red", alpha=0.7)
        # for sf in censored_curves:
        #     plt.step(sf.x, sf(sf.x), where="post", linestyle="-", color="green", alpha=0.7)
        # plt.xlabel("Time")
        # plt.ylabel("Survival Probability")
        # plt.ylim(0, 1)
        # plt.title(f"Survival Function: Non-Survivors (Red) vs. censored (Green)")
        # plt.grid()
        # plt.legend()
        # plt.show()

        c_index = rsf.score(X_test, test_surv_data)
        # c_index = concordance_index_censored(test_surv_data["event"], test_surv_data["time"], risk_scores)[0]
        print(f'c-score for split {split_idx}: {c_index}')
        all_c_scores.append(c_index)
    print(f'Mean c-scores: {np.mean(all_c_scores)}')
    plot_averaged_survival_curves(all_censored_curves, all_event_curves)
    return np.mean(all_c_scores)


def predict_recurrence_classification(splits):
    # model = AdaBoostClassifier(n_estimators=80)
    model = RandomForestClassifier(n_estimators=200,
                                   min_impurity_decrease=0.1,
                                   class_weight={0:1, 1:10})  # Handles class imbalance

    all_labels = []
    all_predictions = []
    for i, (X_train, y_train, X_test, y_test) in enumerate(splits):
        # Fit model to the current split
        model.fit(X_train, y_train['target'])

        # Predict on test set
        y_pred = model.predict(X_test)
        all_predictions.extend(y_pred)
        all_labels.extend(y_test['target'])

    # Evaluate performance
    print(f"=== Split {i + 1} ===")
    print(classification_report(all_labels, all_predictions, target_names=["Stable", "Recurrent"]))
    print(f'Predicted: {all_predictions}')
    print(f'Real:      {np.asarray(all_labels)}')
    print("\n")


if __name__ == "__main__":
    args = get_args()
    df_lesion_metrics = pd.read_csv(args.lesion_metrics_file, header=0, index_col=0)
    df_lesion_metrics['target'] = df_lesion_metrics['MRI_TYPE'].map({'recurrence': 1, 'stable': 0})
    # Get the target for survival analysis
    y = df_lesion_metrics[['target', 'DURATION_TO_IMAG']].copy()

    numeric_columns = df_lesion_metrics.drop(
        columns=['PT_ID', 'LESION_NO', 'SUMMARY_DOSE', 'PATIENT_DIAGNOSIS_PRIMARY',
                 'PATIENT_GENDER', 'MRI_TYPE', 'DURATION_TO_IMAG', 'PATIENT_GENDER', 'PATIENT_DIAGNOSIS_METS',
                 'LESION_COURSE_NO', 'PATIENT_AGE', 'target'])

    # Map a gender to a binary variable
    df_lesion_metrics['gender_id'] = df_lesion_metrics['PATIENT_GENDER'].map({'Male': 1, 'Female': 0})
    # Apply TF-IDF encoding
    vectorizer = TfidfVectorizer(lowercase=True,
                                 stop_words=['brain', 'met', 'mets', 'with', 'ca', 'gk', 'lesions', 'op', 'post'])
    text_features = vectorizer.fit_transform(df_lesion_metrics["PATIENT_DIAGNOSIS_METS"])
    print(vectorizer.get_feature_names_out())
    text_features_df = pd.DataFrame(text_features.toarray(), columns=vectorizer.get_feature_names_out())
    # df_lesion_metrics = pd.concat([df_lesion_metrics.drop(columns=["PATIENT_DIAGNOSIS_METS"]), text_features_df], axis=1)

    if args.pca:
        # pca = decomposition.PCA(n_components=3, whiten=True).fit(numeric_columns)
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(numeric_columns)
        pca = decomposition.PCA()
        df_pca = pca.fit_transform(df_scaled)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= 0.95) + 1
        # if n_components < 20:
        #     n_components = 20
        print(f'PCA selected {n_components} components out of {len(numeric_columns.columns)}')
        selected_components = [f'PC{i}' for i in range(n_components)]
        numeric_columns = pd.DataFrame(df_pca[:, 0:n_components], columns=selected_components)

    X = pd.concat([numeric_columns, df_lesion_metrics[['gender_id', 'SUMMARY_DOSE', 'PATIENT_AGE']]], axis=1)
    # X = pd.concat([numeric_columns, text_features_df, df_lesion_metrics[['gender_id', 'PATIENT_AGE']]], axis=1)

    # top_N_features, top_N_features_gini, top_metrics = select_variables_rf(
    #     X, y, ntrees=20, max_tree_depth=10)

    # Surivval curve averaged across the whole dataset
    plot_survival_curve(y)

    splits = get_random_splits(X, y, 10)
    predict_recurrence_classification(splits)
    #
    # all_c_scores = []
    # for i in range(10):
    #     avg_c_score = predict_recurrence_survival(splits)
    #     all_c_scores.append(avg_c_score)
    # print(f'Final c-score, averaged across runs: mean {np.mean(all_c_scores)}, std. dev {np.std(all_c_scores)}')
    #
