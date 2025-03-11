import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import sksurv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

import torch
import gpytorch
from torch.distributions import Poisson, Uniform
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RBFKernel

from tqdm import tqdm

# Implementation of Inference Algorithm in Fernández et al 2016 "Gaussian Processes for Survival Analysis"


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
        # TODO work out RBF kernel to have interaction form K((t_1,X_1),(t_2,X_2)) := K_0(s,t) + \sum_i X_1i X_2i K_i(s,t)
        # Base K(t,s) Kernel
        self.time_kernel = RBFKernel(active_dims=[0])
        self.covar_module = self.time_kernel

        # All interaction term Kernels K(X_i, Y_i) * K(t_i,s_i)
        for i in range(num_covariates):
            covar_kernel = ScaleKernel(RBFKernel(active_dims=[i+1]))
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
beta = 4
alpha = 0.5

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
def inference(T, X, targets=None, beta=beta, alpha=alpha, num_aug=5, num_epochs=1_000):
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
        # TODO When we are dealing with right censored, data, this will be a mixture of 1s and 0s depending on whether censored
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
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(inputs)
        loss = -mll(output, targets)
        loss.backward()
        if (epoch+1) % (num_epochs // 10) == 0:
            print(f"Initial Train: Epoch {epoch+1}/{num_epochs} - Loss: {loss.item()}")
        optimizer.step()

    # t_new = torch.Tensor([10.0])
    # x_new = torch.tensor([[0.5, 0.5]])
    # new_input = torch.cat([t_new.unsqueeze(0), x_new], dim=1)
    # print(new_input)

    # pred = likelihood(model_init(new_input))
    # print(pred.mean.item())
    # print(pred.variance.item())

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
            n_i = Poisson(Lambda_t).sample().long().item() + 1
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
            # Also add the observed event time with label 1:
            # candidate_times_list.append(t_i.unsqueeze(0))
            # candidate_labels_list.append(torch.ones(1))
            # candidate_X_list.append(x_i)

        # TODO Refactor this training process, this is ugly code
        # Update GP params with the rejected datapoints
        X_n = torch.stack(candidate_X_list)
        T_n = torch.stack(candidate_times_list)
        targets_n = torch.stack(candidate_labels_list).squeeze(1)
        all_times = torch.cat((all_times, T_n))   # (N_aug,)
        all_labels = torch.cat((all_labels, targets_n), dim=0)   # (N_aug,)
        all_X = torch.cat((all_X, X_n), dim=0)             # (N_aug, d)

        inputs = torch.column_stack((all_times, all_X))
        num_inducing = max(30, N//5)
        inducing_idx = torch.randperm(inputs.shape[0])[:num_inducing]
        inducing_points = inputs[inducing_idx]

        model.train()
        likelihood.train()

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = model(inputs)
            loss = -mll(output, all_labels)
            loss.backward()
            if (epoch+1) % (num_epochs // 10) == 0:
                print(f"Augmnentation {n+1} Train: Epoch {epoch+1}/{num_epochs} - Loss: {loss.item()}")
            optimizer.step()
        
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

def plot_survival(model, likelihood):
    # For now, we use simple approximation S(t|x) \approx \exp(-\Gamma_0(t) * \sigma(l(t,x)))
    return None

def plot_hazard(model, likelihood, X, samps=5):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Sample from covariates of three Individuals
    T_new = torch.linspace(0, 50, 50)
    unique_ids = X['PT_ID'].unique()
    sample_ids = np.random.choice(unique_ids, size=samps, replace=False)

    grouped = X[X['PT_ID'].isin(sample_ids)].groupby('PT_ID')
    samples = grouped.apply(lambda group: group.sample(n=1)).reset_index(drop=True)
    samples = samples.drop(columns=['PT_ID'])
    samples = torch.Tensor(samples.values)        

    model.eval()
    likelihood.eval()

    # Run through timestep of each sample, showing its hazard function
    for i, sample in enumerate(samples):
        samp_steps_mean = []
        samp_steps_var = []
        base_hazard = []
        for t_new in T_new:
            new_input = torch.cat((t_new.unsqueeze(-1), sample),dim=-1).unsqueeze(0)
            with torch.no_grad():
                pred_samp_t = likelihood(model(new_input))
            # Extract both mean an variance of GP, remember we structure this a Base Hazard * GP Hazard
            samp_steps_mean.append(pred_samp_t.mean.item() * lambda0(t_new.numpy()))
            samp_steps_var.append(pred_samp_t.variance.item() * lambda0(t_new.numpy()))
            base_hazard.append(lambda0(t_new.numpy()))
        ax.plot(np.linspace(0, 50, 50), samp_steps_mean, label=f'ID: {sample_ids[i]}')
        ax.fill_between(np.linspace(0, 50, 50), lambda0(t_new.numpy()) * samp_steps_var 
                        + samp_steps_mean, samp_steps_mean - samp_steps_var * lambda0(t_new.numpy()))
        ax.plot(np.linspace(0, 50, 50), base_hazard)

    ax.set_title('Estimated Hazard Functions using GP Model')
    ax.set_xlabel('time')
    ax.set_ylabel('Hazard')
    plt.legend()
    plt.grid()
    plt.savefig('./survival_pred/Hazard_Plot.pdf')
    return None

def predict_recurrence_survival():
    return None

def split_data(X, T, target, train_ratio=0.9, seed=None):
    """
    Splits input tensors (X, T, target) into training and testing sets
    """    
    # Ensure all tensors have the same number of samples
    num_samples = X.size(0)
    
    shuffled_indices = torch.randperm(num_samples)
    
    # Split indices into training and testing
    split_idx = int(num_samples * train_ratio)
    train_indices = shuffled_indices[:split_idx]
    test_indices = shuffled_indices[split_idx:]

    X_train, X_test = X[train_indices], X[test_indices]
    T_train, T_test = T[train_indices], T[test_indices]
    target_train, target_test = target[train_indices], target[test_indices]

    test_has_positive = (target_train == 1).any()
    
    # if there is no recurrence in training data, reshuffle until there is
    while not test_has_positive:
        shuffled_indices = torch.randperm(num_samples)

        split_idx = int(num_samples * train_ratio)
        train_indices = shuffled_indices[:split_idx]
        test_indices = shuffled_indices[split_idx:]

        X_train, X_test = X[train_indices], X[test_indices]
        T_train, T_test = T[train_indices], T[test_indices]
        target_train, target_test = target[train_indices], target[test_indices]

        test_has_positive = (target_train == 1).any()
        
    return X_train, X_test, T_train, T_test, target_train, target_test

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

# TODO Build Data pipeline to feed into Inference Algorithm
# need to split our patient data into (Targets, T, X) where Targets are indicator of recurrance, T
# is augmented to be time since first image or last recurrance, and X is the matrix of all other covariates
if __name__ == "__main__":
    # Toy Example
    # # N patients, with survival times T, two covariates uniformly disted
    # N = 50
    # T = torch.linspace(0.5, 5.0, N)
    # X = torch.rand(N, 2)
    
    # # Train the GP classifier using our data augmentation scheme
    # model, likelihood, cand_times, cand_labels, cand_X = inference(T, X, num_epochs=100)
    
    # # Simplest Possible Inference - single data point
    # # To predict the probability of an event (i.e. the acceptance probability \sigma(l(t,x)))
    # # at a new time t_new and covariate x_new, we form an input and query the GP
    # model.eval()
    # likelihood.eval()
    # t_new = torch.Tensor([3.0])
    # x_new = torch.tensor([[0.5, 0.5]])
    # new_input = torch.cat([t_new.unsqueeze(0), x_new], dim=1)
    
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     pred = likelihood(model(new_input))
    #     # pred.mean is our estimate for \sigma(l(t,x)).
    #     print(f"Predicted acceptance probability (event) at t={t_new.flatten()}:", pred.mean.item())
    
    #     # To obtain a survival probability, one would typically combine this with the baseline hazard
    #     # For example, one may compute: S(t|x) \approx \exp(-\Gamma_0(t) * \sigma(l(t,x)))
    #     S_t = torch.exp(-Lambda0(t_new) * pred.mean)
    #     print(f"Estimated survival probability at t={t_new.flatten()}:", S_t.item())
    #     print(f"Estimated variance at t={t_new.flatten()}:", 
    #           torch.exp(-Lambda0(t_new) * pred.variance).item())

    # Load in Patient Metrics Data
    lesion_metrics_file = 'radiomics_metrics.csv'
    df_lesion_metrics = pd.read_csv(lesion_metrics_file, header=0, index_col=0)
    # Map target and gender_id to binary indicators
    df_lesion_metrics['target'] = df_lesion_metrics['MRI_TYPE'].map({'recurrence': 1, 'stable': 0})
    df_lesion_metrics['gender_id'] = df_lesion_metrics['PATIENT_GENDER'].map({'Male': 1, 'Female': 0})

    # Adjust time representation so that DUR_TO_IMAGE represenets the time since either last recurrance
    # or measurements have started. Allows us to represent non-recurrant data as right censored data 
    # Of course there are dubious longitudinal aspects of such an approach that may need to be addressed
    df_lesion_metrics = df_lesion_metrics.sort_values(['PT_ID', 'DURATION_TO_IMAG'])
    df_lesion_metrics['group'] = df_lesion_metrics.groupby('PT_ID')['target'].cumsum().shift(1).fillna(0).astype(int)
    df_lesion_metrics['time'] = df_lesion_metrics.groupby(['PT_ID', 'group'])['DURATION_TO_IMAG'].cumsum()

    target = torch.Tensor(df_lesion_metrics[['target']].copy().values).squeeze(1)
    T = torch.Tensor(df_lesion_metrics[['time']].copy().values)

    numeric_columns_w_ID = df_lesion_metrics.drop(
        columns=['LESION_NO', 'PATIENT_DIAGNOSIS_PRIMARY',
                'PATIENT_GENDER', 'MRI_TYPE', 'DURATION_TO_IMAG', 'PATIENT_GENDER', 'PATIENT_DIAGNOSIS_METS',
                'LESION_COURSE_NO', 'PATIENT_AGE', 'target', 'group', 'time'])
        
    numeric_columns = numeric_columns_w_ID.drop(columns=['PT_ID'])
    
    # # As per Alex's idea, want to apply TF-IDF encoding to the text diagnosis data
    # vectorizer = TfidfVectorizer(lowercase=True, stop_words=['brain','met','mets','with', 'ca', 'gk', 'lesions','op', 'post'])
    # text_features = vectorizer.fit_transform(df_lesion_metrics["PATIENT_DIAGNOSIS_METS"])
    # text_features_df = pd.DataFrame(text_features.toarray(), columns=vectorizer.get_feature_names_out())

    X = torch.Tensor(numeric_columns.values)

    # top_N_features, top_N_featuers_gini, top_metrics = select_variables_rf(
    #     pd.DataFrame(torch.cat((X, T),dim=1).numpy()), pd.DataFrame(target.unsqueeze(1).numpy()),
    #     ntrees=20, max_tree_depth=10
    # )

    X_train, X_test, T_train, T_test, target_train, target_test = split_data(X, T, target, 
                                                                             train_ratio=0.9, seed=None)

    model, likelihood, _, _, _ = inference(T_train, X_train, target_train, num_epochs= 10, num_aug=1)

    plot_hazard(model, likelihood, numeric_columns_w_ID)