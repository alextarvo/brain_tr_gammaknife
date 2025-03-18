import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

import torch
import gpytorch
from torch.distributions import Poisson, Uniform
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import PolynomialKernel
from gpytorch.kernels import RBFKernel

from tqdm import tqdm

# Implementation of Inference Algorithm in Fernández et al 2016 "Gaussian Processes for Survival Analysis"

# Parameters used in testing
N_FOLDS = 10
N_EPOCH = 1_000
N_AUG = 0

# GP CLASSIFICATION MODEL
# We define a variational GP model to classify points as "accepted" (observed event) or "rejected"
# NOTE - Requires Zero mean and stationary Kernel for Survival Function properties to be ensured
class SurvivalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_covariates):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        # Base K(t,s) Kernel
        self.time_kernel = ScaleKernel(PolynomialKernel(active_dims=[0], power=1))
        self.covar_module = self.time_kernel

        # All interaction term Kernels K(X_i, Y_i) * K(t_i,s_i)
        for i in range(num_covariates):
            # covar_kernel = ScaleKernel(PolynomialKernel(active_dims=[i+1], power=1))
            covar_kernel = ScaleKernel(RBFKernel(active_dims=[i+1], power=1))
            interaction = covar_kernel * self.time_kernel
            self.covar_module += interaction
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Since this is a classification model, w Sigmoid Link, need to use Bernoulli Likelihood
likelihood = gpytorch.likelihoods.BernoulliLikelihood()

# Here we use a Weibull baseline hazard: \lambda_0(t) = 2\beta t^(\alpha-1)
# For now, we will manually fiddle with these hyperparams, maybe use some random search
# TODO Perform Guassian Analysis (MCMC) - paper recommends Gamma prior on \beta, Unif(0,2.3) on alpha, step at implement at Augmentation
beta = 2
alpha = 1.1

def lambda0(t, beta=beta, alpha=alpha):
    # We use the Weibull base Hazard, as it is fairly standard in application 
    return 2 * beta * t**(alpha - 1)

def Lambda0(T, beta=beta, alpha=alpha):
    # Cumulative hazard - tractable integral when choosing Weibull base hazard
    return 2 * beta / alpha * T**alpha

def Lambda0_inv(u, beta=beta, alpha=alpha):
    # Inverse of \Gamma_0(t)
    return (alpha * u / (2 * beta))**(1 / alpha)


# DATA INFERENCE ALGO
# For each subject with observed time T_i and covariate X_i,
# sample candidate points from a Poisson process with rate \Gamma_0(T_i) and then transform them.
def inference(T, X, targets, T_test, X_test, targets_test,
               beta=beta, alpha=alpha, num_aug=5, num_epochs=1_000):
    '''
    T: (N,) tensor of survival times.
    X: (N, d) tensor of covariates.
    We will build an augmented dataset: for each subject,
    the observed event (label 1) and candidate (rejected) points (label 0).'
    '''
    # Train on original dataset before performing MCMC with Augmented Data
    # For variational GP, choose inducing points from the data
    inputs = torch.column_stack((T, X))
    if targets is None:
        targets = torch.ones(inputs.shape[0])
    num_inducing = min(10, inputs.shape[0])
    inducing_idx = torch.randperm(inputs.shape[0])[:num_inducing]
    inducing_points = inputs[inducing_idx]

    model = SurvivalGPModel(inducing_points, X.shape[1])
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=inputs.shape[0])

    # Targets are defined as interactions between X and T - these should all be classified as accepted jumps
    # TODO Implement a dataloader for batch GD - for the dataset of our size we're prob fine w/o it, but good practice
    train_message = []
    for epoch in range(num_epochs):
        likelihood.train()
        optimizer.zero_grad()
        output = model(inputs)
        loss = -mll(output, targets)
        loss.backward()
        if (epoch+1) % (num_epochs // 10) == 0:
            likelihood.eval()
            model.eval()

            # Calculate C-Indexes in training
            with torch.no_grad():                
                test = model(torch.column_stack([T_test, X_test]))
                prediction_test = likelihood(test)

                train = model(torch.column_stack([T_train, X_train]))
                prediction_train = likelihood(train)

            hazard_est_test = prediction_test.mean
            c_test = concordance_index_censored(target_test.bool(),
                                            T_test.squeeze(1), hazard_est_test)
            print(f'Initial Test: Epoch {epoch+1}/{num_epochs} - C-Index: {c_test[0]}')
            # Make the training and test losses more readable, seperate them out by putting training
            # in an array and printing at the end
            hazard_est_train = prediction_train.mean
            c_train = concordance_index_censored(target_train.bool(),
                                            T_train.squeeze(1), hazard_est_train)

            train_message.append(f'Inital Train: Epoch {epoch+1}/{num_epochs} - C-Index: {c_train[0]}')
        optimizer.step()

    for result in train_message: print(result)

    all_times = T
    all_labels = torch.ones(inputs.shape[0])
    all_X = X

    candidate_times_list = []
    candidate_labels_list = []
    candidate_X_list = []

    # Let's only run the augmentation loop for data with recurrence
    full_data = torch.column_stack((T, X, targets))
    recurrent_data = full_data[full_data[:, -1] == 1]

    recurrent_T = recurrent_data[:,0]
    recurrent_X = recurrent_data[:,1:-1]
    N = recurrent_T.shape[0]


    for n in range(num_aug):
        for i in tqdm(range(N)):
            t_i = recurrent_T[i]
            x_i = recurrent_X[i]
            # Calculate cumulative hazard at t_i:
            Lambda_t = Lambda0(t_i, beta, alpha)
            # Sample number of candidate points from Poisson(\Gamma_0(t_i))
            n_i = (Poisson(Lambda_t).sample().long().item() + 1)//10
            # Sample n_i points uniformly on [0, \Gamma_0(t_i)]
            u = Uniform(0, Lambda_t).sample((n_i,))
            # Map these to candidate times via the inverse cumulative hazard:
            t_candidates = Lambda0_inv(u, beta, alpha)
            print(f'\n Time Sampled: {t_i}, Num Rejects: {n_i}')
            for t_cand in t_candidates:
                new_input = torch.cat((t_cand.unsqueeze(-1), x_i),dim=-1).unsqueeze(0)
                likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    pred_cand = likelihood(model(new_input))
                u_cand = Uniform(0,1).sample().item()
                if u_cand < 1 - pred_cand.mean.item():
                    candidate_times_list.append(t_cand.unsqueeze(0))
                    candidate_labels_list.append(torch.zeros(1))  # rejected (label 0)
                    candidate_X_list.append(x_i)

        # TODO Refactor this training process, this is ugly code
        # Update GP params with the rejected datapoints
        X_n = torch.stack(candidate_X_list)
        T_n = torch.stack(candidate_times_list)
        targets_n = torch.stack(candidate_labels_list).squeeze(1)
        all_times = torch.cat((all_times, T_n))   # (N_aug,)
        all_labels = torch.cat((all_labels, targets_n), dim=0)   # (N_aug,)
        all_X = torch.cat((all_X, X_n), dim=0)             # (N_aug, d)

        inputs = torch.column_stack((all_times, all_X))
        num_inducing = N//5
        inducing_idx = torch.randperm(inputs.shape[0])[:num_inducing]
        inducing_points = inputs[inducing_idx]

        model.train()
        likelihood.train()

        test_message = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = model(inputs)
            loss = -mll(output, all_labels)
            loss.backward()
            if (epoch+1) % (num_epochs // 10) == 0:
                print(f"Augmnentation {n+1} Train: Epoch {epoch+1}/{num_epochs} - Loss: {loss.item()}")
                if X_test is not None and T_test is not None and target_test is not None:
                    output_test = model(torch.column_stack([T_test, X_test]))
                    loss_test = -mll(output_test, targets_test)
                    test_message.append(f'Augmentation {n+1} Tets: Epoch {epoch+1}/{num_epochs} - Loss: {loss_test.item()}')

            optimizer.step()
        for result in test_message: print(result)
        
    return model, likelihood, all_times, all_labels, all_X


# # Trainnig GP - Since we non-normal likelihood, cannot use ExactGP, use inducing points 
# # Variational Approx of Posterior
# def train_survival_gp(T, X, num_epochs=1_000):
#     # Augment the data:
#     aug_times, aug_X, aug_labels = inference(T, X, num_aug=1)
#     # Combine time and covariates into a single input feature: [t, x_1, ..., x_d]
#     inputs = torch.cat([aug_times.unsqueeze(1), aug_X], dim=1)
#     targets = aug_labels  # 1 for observed event, 0 for candidate (rejected) points
    
#     # For variational GP, choose inducing points from the data
#     num_inducing = min(30, inputs.shape[0])
#     inducing_idx = torch.randperm(inputs.shape[0])[:num_inducing]
#     inducing_points = inputs[inducing_idx]
    
#     model = SurvivalGPModel(inducing_points, X.shape[1])
#     model.train()
#     likelihood.train()
    
#     # TODO Implement Bayesian Optimization with Pybo
#     optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)
#     mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=inputs.shape[0])
    
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         output = model(inputs)
#         loss = -mll(output, targets)
#         loss.backward()
#         if (epoch+1) % (num_epochs // 10) == 0:
#             print(f"Initial Train: Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.3f}")
#         optimizer.step()
#     return model, likelihood

def plot_hazard(model, likelihood, X, samps=3):
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Sample from covariates of three Individuals
    T_new = torch.linspace(0, 50, 50)
    unique_ids = X['PT_ID'].unique()
    sample_ids = np.random.choice(unique_ids, size=samps, replace=False)

    # Sampling from the datapoints of each ID
    grouped = X[X['PT_ID'].isin(sample_ids)].groupby('PT_ID')
    samples = grouped.apply(lambda group: group.sample(n=1)).reset_index(drop=True)
    samples = samples.drop(columns=['PT_ID'])
    samples = torch.Tensor(samples.values)        

    model.eval()
    likelihood.eval()

    # Run through timestep of each sample, showing its hazard function
    for i, sample in enumerate(samples):
        x_ax = np.linspace(0.1, 50, 50)
        colors = ['blue', 'green', 'orange']
        samp_steps_mean = np.empty((50))
        samp_steps_var = np.empty((50))
        base_hazard = np.empty((50))
        for j, t_new in enumerate(T_new):
            new_input = torch.cat((t_new.unsqueeze(-1), sample),dim=-1).unsqueeze(0)
            with torch.no_grad():
                pred_samp_t = likelihood(model(new_input))
            # Extract both mean an variance of GP, remember we structure this a Base Hazard * GP Hazard
            samp_steps_mean[i] = (pred_samp_t.mean.item() * lambda0(t_new.numpy()))
            samp_steps_var[i] = (pred_samp_t.variance.item() * lambda0(t_new.numpy()))
            base_hazard[i] = (lambda0(t_new.numpy()))
        ax1.plot(x_ax, samp_steps_mean, label=f'ID: {sample_ids[i]}', c=colors[i])
        ax1.fill_between(x_ax, samp_steps_var/2 + samp_steps_mean, 
                        samp_steps_mean - samp_steps_var/2, color=colors[i], alpha=0.2)
        # ax.plot(x_ax, base_hazard)

        # To obtain a survival probability, one would typically combine this with the baseline hazard
        # For example, one may compute: S(t|x) \approx \exp(-\Gamma_0(t) * \sigma(l(t,x)))
        ax2.plot(x_ax, np.ones(50) - np.exp(-samp_steps_mean), label=f'ID: {sample_ids[i]}', c=colors[i])
        ax2.fill_between(x_ax, np.ones(50) - np.exp(-(samp_steps_mean + samp_steps_var)), 
                 np.ones(50) - np.exp(-(samp_steps_mean - samp_steps_var)), color=colors[i], alpha=0.2)

    ax1.set_title('Estimated Hazard Functions using GP Model')
    ax1.set_xlabel('time')
    ax1.set_ylabel('Hazard')
    ax1.set_ylim([-10,100])
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Estimated Survival Functions using GP Model')
    ax2.set_xlabel('time')
    ax2.set_ylabel('Conditional Cumulative Survival')
    ax2.set_ylim([-0.5, 1.5])
    ax2.legend()
    ax2.grid(True)

    fig1.savefig('./survival_pred/Hazard_Plot.pdf')
    fig2.savefig('./survival_pred/Survival_Plot.pdf')
    return None

def split_data_indx(X, T, target, splits=5, seed=None):
    """
    Splits input tensors (X, T, target) into training and testing sets
    """    
    # Ensure all tensors have the same number of samples
    num_samples = X.size(0)
    skf = StratifiedKFold(n_splits=splits, shuffle=True)
    k_split = []
    
    # Split indices into training and testing - ensured these folds are balanced
    # because of StratifiedKFold
    for train, test in skf.split(X, target):
        k_split.append((X[train], T[train], target[train],
                        X[test], T[test], target[test]))
    return k_split

def select_variables_rf(X, y, ntrees, max_tree_depth, num_features=20,visualize=False):
    """
    Alex's Approach for choosing top variables with a RF classifier
    """
    model = RandomForestClassifier(
        n_estimators=ntrees,
        max_depth=max_tree_depth).fit(X, y)

    metrics_gini = model.feature_importances_
    
    predictor_columns = X.columns.tolist()
    assert (len(predictor_columns) == len(metrics_gini))
    features_importance_list = list(zip(metrics_gini, predictor_columns))
    features_importance_list = sorted(features_importance_list, key=lambda x: abs(x[0]), reverse=True)
    gini_coefficients_sorted, feature_names_sorted = zip(*features_importance_list)

    top_N_features = feature_names_sorted[0:num_features]
    top_N_features_gini = gini_coefficients_sorted[0:num_features]
    top_metrics = X[list(top_N_features)]

    return top_N_features, top_N_features_gini, top_metrics

# need to split our patient data into (Targets, T, X) where Targets are indicator of recurrance, T
# is augmented to be time since first image or last recurrance, and X is the matrix of all other covariates
if __name__ == "__main__":
    # Load in Patient Metrics Data
    lesion_metrics_file = 'radiomics_metrics_pyradiomics.csv'
    df_lesion_metrics = pd.read_csv(lesion_metrics_file, header=0, index_col=0)
    # Map target and gender_id to binary indicators
    df_lesion_metrics['target'] = df_lesion_metrics['MRI_TYPE'].map({'recurrence': 1, 'stable': 0})
    df_lesion_metrics['gender_id'] = df_lesion_metrics['PATIENT_GENDER'].map({'Male': 1, 'Female': 0})

    target = torch.Tensor(df_lesion_metrics[['target']].copy().values).squeeze(1)
    T = torch.Tensor(df_lesion_metrics[['DURATION_TO_IMAG']].copy().values)

    numeric_columns = df_lesion_metrics.drop(
        columns=['PT_ID', 'LESION_NO', 'PATIENT_DIAGNOSIS_PRIMARY',
                'PATIENT_GENDER', 'MRI_TYPE', 'DURATION_TO_IMAG', 'PATIENT_DIAGNOSIS_METS',
                'target'
                ])
        
    # Perform PCA on the features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(numeric_columns)
    pca = decomposition.PCA()
    df_pca = pca.fit_transform(df_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f'PCA selected {n_components} components out of {len(numeric_columns.columns)}')
    selected_components = [f'PC{i}' for i in range(n_components)]
    numeric_columns = pd.DataFrame(df_pca[:, 0:n_components], columns=selected_components)

    
    # # As per Alex's idea, want to apply TF-IDF encoding to the text diagnosis data
    # vectorizer = TfidfVectorizer(lowercase=True, stop_words=['brain','met','mets','with', 'ca', 'gk', 'lesions','op', 'post'])
    # text_features = vectorizer.fit_transform(df_lesion_metrics["PATIENT_DIAGNOSIS_METS"])
    # text_features_df = pd.DataFrame(text_features.toarray(), columns=vectorizer.get_feature_names_out())

    X = torch.Tensor(numeric_columns.values)

    # top_N_features, top_N_featuers_gini, top_metrics = select_variables_rf(
    #     pd.DataFrame(torch.cat((X, T),dim=1).numpy()), pd.DataFrame(target.unsqueeze(1).numpy()),
    #     ntrees=20, max_tree_depth=10
    # )

    # Perform cross-validation on our data
    C_index = []
    k_indx = split_data_indx(X, T, target, splits=N_FOLDS, seed=None)

    for fold, (X_train, T_train, target_train, X_test, T_test, target_test) in enumerate(k_indx):

        model, likelihood, _, _, _ = inference(T_train, X_train, target_train,
                                            T_test, X_test, target_test,
                                                num_epochs=N_EPOCH, num_aug=N_AUG)

        if fold == len(k_indx):
            # Need ID for plotting purposes
            numeric_columns['PT_ID'] = df_lesion_metrics['PT_ID']
            plot_hazard(model, likelihood, numeric_columns)

        print(f'\n --------------FOLD {fold}------------------\n')

        # Get the C-Index on the testing set - want to use time dependent version
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test = torch.column_stack([T_test, X_test])
            test_pred = likelihood(model(test))
        
        # Now compute the Hazard at testing points and then C-Index
        # hazard_est = lambda0(T_test.squeeze(1)) * test_pred.mean
        hazard_est = test_pred.mean # * lambda0(T_test.squeeze(1))
        print('Hazard Estimation: \n',hazard_est, '\n Reccurence Values: \n', 
            target_test.bool(), '\n Time to Image: \n', T_test.squeeze(1))
        c_index_test = concordance_index_censored(target_test.bool(),
                                            T_test.squeeze(1), hazard_est)
        C_index.append(c_index_test[0])
        print(f'\n C-Index over Test for Fold {fold}:  {c_index_test}')

        # Do the same with training data - to see the model was at least fitting training
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            training = torch.column_stack([T_train, X_train])
            train_pred = likelihood(model(training))
        
        hazard_est_train = train_pred.mean # * lambda0(T_train.squeeze(1))
        c_index_train = concordance_index_censored(target_train.bool(),
                                            T_train.squeeze(1), hazard_est_train)
        print(f'\n C-Index over Train for Fold {fold}: {c_index_train}')

        # Loss of Test data
        test_loss = gpytorch.metrics.mean_squared_error(test_pred, target_test)
        print(f'\nTest Loss on Fold {fold} (Bern Likelihood)\n: {test_loss}')
    
    print('\n----------------------Final Results-------------------------\n')
    print(f'Average C-index across {N_FOLDS}-Folds: {np.average(C_index)}')