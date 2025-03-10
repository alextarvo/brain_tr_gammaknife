import torch
import gpytorch
from torch.distributions import Poisson, Uniform
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RBFKernel

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
# TODO Perform Guassian Analysis (MCMC) - paper recommends Gamma prior on \beta, Unif(0,2.3) on alpha - implement at Augmentation step
beta = 1.0
alpha = 1.5

def lambda0(t, beta=beta, alpha=alpha):
    # We use the Weibull base Hazard, as it is fairly standard in application 
    return 2 * beta * t**(alpha - 1)

def Lambda0(T, beta=beta, alpha=alpha):
    # Cumulative hazard - tractable integral when choosing Weibull base hazard
    return 2 * beta / alpha * T**alpha

def Lambda0_inv(u, beta=beta, alpha=alpha):
    # Inverse of \Gamma_0(t)
    return (alpha * u / (2 * beta))**(1 / alpha)

def logit(x):
    return torch.logit(x)

# DATA INFERENCE ALGO
# For each subject with observed time T_i and covariate X_i,
# sample candidate points from a Poisson process with rate \Gamma_0(T_i) and then transform them.
def inference(T, X, beta=beta, alpha=alpha, num_aug=10, num_epochs=1_000):
    '''
    T: (N,) tensor of survival times.
    X: (N, d) tensor of covariates.
    We will build an augmented dataset: for each subject,
    the observed event (label 1) and candidate (rejected) points (label 0).'
    '''
    # Train on original dataset before performing MCMC with Augmented Data
    # For variational GP, choose inducing points from the data
    inputs = torch.column_stack((T, X))
    # TODO When we are dealing with right censored, data, this will be a mixture of 1s and 0s depending on whether censored
    targets = torch.ones(inputs.shape[0])
    num_inducing = min(30, inputs.shape[0])
    inducing_idx = torch.randperm(inputs.shape[0])[:num_inducing]
    inducing_points = inputs[inducing_idx]
    N = T.shape[0]

    model = SurvivalGPModel(inducing_points, X.shape[1])
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=inputs.shape[0])

    # Targets are defined as interactions between X and T - these should all be classified as accepted jumps
    # TODO Implement a dataloader for batch GP - for the dataset of our size we're prob fine wo it, but good practice
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

    all_times = T.unsqueeze(1)
    all_labels = torch.ones(N)
    all_X = X

    candidate_times_list = []
    candidate_labels_list = []
    candidate_X_list = []

    for n in range(num_aug):
        for i in range(N):
            t_i = T[i]
            x_i = X[i]
            # Calculate cumulative hazard at t_i:
            Lambda_t = Lambda0(t_i, beta, alpha)
            # Sample number of candidate points from Poisson(\Gamma_0(t_i))
            n_i = Poisson(Lambda_t).sample().long().item() + 1
            # Sample n_i points uniformly on [0, \Gamma_0(t_i)]
            u = Uniform(0, Lambda_t).sample((n_i,))
            # Map these to candidate times via the inverse cumulative hazard:
            t_candidates = Lambda0_inv(u, beta, alpha)
            for t_cand in t_candidates:
                new_input = torch.cat((t_cand.unsqueeze(0), x_i),dim=0).unsqueeze(0)
                likelihood.eval()
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
        all_times = torch.cat((all_times, T_n), dim=0)   # (N_aug,)
        all_labels = torch.cat((all_labels, targets_n), dim=0)   # (N_aug,)
        all_X = torch.cat((all_X, X_n), dim=0)             # (N_aug, d)

        inputs = torch.column_stack((all_times, all_X))
        num_inducing = min(30, inputs.shape[0])
        inducing_idx = torch.randperm(inputs.shape[0])[:num_inducing]
        inducing_points = inputs[inducing_idx]

        print(inputs.shape, all_labels.shape)

        model.train()
        likelihood.train()

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = model(inputs)
            loss = -mll(output, all_labels)
            loss.backward()
            if (epoch+1) % (num_epochs // 10) == 0:
                print(f"Augmnentation {n} Train: Epoch {epoch+1}/{num_epochs} - Loss: {loss.item()}")
            optimizer.step()
        
    return model, likelihood


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

# Toy example
# TODO Build Data pipeline to feed into Inference Algorithm
# need to split our patient data 
if __name__ == "__main__":
    # More complex to data, N patients, with survival times T, two covariates uniformly disted
    N = 50
    T = torch.linspace(0.5, 5.0, N)
    X = torch.rand(N, 2)
    
    # Train the GP classifier using our data augmentation scheme
    model, likelihood = inference(T, X, num_epochs=100)
    
    # Simplest Possible Inference - single data point
    # To predict the probability of an event (i.e. the acceptance probability \sigma(l(t,x)))
    # at a new time t_new and covariate x_new, we form an input and query the GP
    model.eval()
    likelihood.eval()
    t_new = torch.Tensor([10.0])
    x_new = torch.tensor([[0.5, 0.5]])
    new_input = torch.cat([t_new.unsqueeze(0), x_new], dim=1)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(new_input))
        # pred.mean is our estimate for \sigma(l(t,x)).
        print(f"Predicted acceptance probability (event) at t={t_new.flatten()}:", pred.mean.item())
    
        # To obtain a survival probability, one would typically combine this with the baseline hazard
        # For example, one may compute: S(t|x) \approx \exp(-\Gamma_0(t) * \sigma(l(t,x)))
        S_t = torch.exp(-Lambda0(t_new) * pred.mean)
        print(f"Estimated survival probability at t={t_new.flatten()}:", S_t.item())