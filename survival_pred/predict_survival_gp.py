import torch
import gpytorch
from torch.distributions import Poisson, Uniform
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import VariationalDistribution
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RBFKernel

# Implementation of Inference Algorithm in Fernández et al 2016 "Gaussian Processes for Survival Analysis"

# Here we use a Weibull baseline hazard: \lambda_0(t) = 2\beta t^(\alpha-1)
# For now, we will manually fiddle with these hyperparams, maybe use some random search
# TODO Perform Guassian Analysis (MCMC) - paper recommends Gamma prior on \beta, Unif(0,2.3) on alpha
beta = 1.0
alpha = 1.0

def lambda0(t, beta=beta, alpha=alpha):
    return 2 * beta * t**(alpha - 1)

def Lambda0(t, beta=beta, alpha=alpha):
    # Cumulative hazard - tractable integral when choosing Weibull base hazard
    return 2 * beta / alpha * t**alpha

def Lambda0_inv(u, beta=beta, alpha=alpha):
    # Inverse of \Gamma_0(t)
    return (alpha * u / (2 * beta))**(1 / alpha)

def sigma(x):
    return torch.sigmoid(x)

# DATA AUGMENTATION
# For each subject with observed time T_i and covariate X_i,
# sample candidate points from a Poisson process with rate \Gamma_0(T_i) and then transform them.
def augment_data(T, X, beta=beta, alpha=alpha):
    '''
    T: (N,) tensor of survival times.
    X: (N, d) tensor of covariates.
    We will build an augmented dataset: for each subject,
    the observed event (label 1) and candidate (rejected) points (label 0).'
    '''
    candidate_times_list = []
    candidate_labels_list = []
    candidate_X_list = []
    
    N = T.shape[0]
    for i in range(N):
        t_i = T[i]
        x_i = X[i]
        # Calculate cumulative hazard at t_i:
        Lambda_t = Lambda0(t_i, beta, alpha)
        # Sample number of candidate points from Poisson(\Gamma_0(t_i))
        n_i = Poisson(Lambda_t).sample().long().item()
        if n_i > 0:
            # Sample n_i points uniformly on [0, \Gamma_0(t_i)]
            u = Uniform(0, Lambda_t).sample((n_i,))
            # Map these to candidate times via the inverse cumulative hazard:
            t_candidates = Lambda0_inv(u, beta, alpha)
            candidate_times_list.append(t_candidates)
            candidate_labels_list.append(torch.zeros(n_i))  # rejected (label 0)
            candidate_X_list.append(x_i.repeat(n_i, 1))
        # Also add the observed event time with label 1:
        candidate_times_list.append(t_i.unsqueeze(0))
        candidate_labels_list.append(torch.ones(1))
        candidate_X_list.append(x_i.unsqueeze(0))
    
    all_times = torch.cat(candidate_times_list)   # (N_aug,)
    all_labels = torch.cat(candidate_labels_list)   # (N_aug,)
    all_X = torch.cat(candidate_X_list)             # (N_aug, d)
    return all_times, all_X, all_labels

# GP CLASSIFICATION MODEL
# We define a variational GP model to classify points as "accepted" (observed event) or "rejected"
# NOTE - Requires Zero mean and stationary Kernel for Survival Function properties to be ensured
class SurvivalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = VariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        # TODO work out RBF kernel to have interaction form K((t_1,X_1),(t_2,X_2)) := K_0(s,t) + \sum_i X_1i X_2i K_i(s,t)
        self.covar_module = ScaleKernel(RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Since this is a classification model, w Sigmoid Link, need to use Bernoulli Likelihood
likelihood = gpytorch.likelihoods.BernoulliLikelihood()

# Trainnig GP - Since we non-normal likelihood, cannot use ExactGP, use inducing points 
# Variational Approx of Posterior
def train_survival_gp(T, X, num_epochs=1_000):
    # Augment the data:
    aug_times, aug_X, aug_labels = augment_data(T, X)
    # Combine time and covariates into a single input feature: [t, x_1, ..., x_d]
    inputs = torch.cat([aug_times.unsqueeze(1), aug_X], dim=1)
    targets = aug_labels  # 1 for observed event, 0 for candidate (rejected) points
    
    # For variational GP, choose inducing points (here, 20 random ones).
    num_inducing = min(20, inputs.shape[0])
    inducing_idx = torch.randperm(inputs.shape[0])[:num_inducing]
    inducing_points = inputs[inducing_idx]
    
    model = SurvivalGPModel(inducing_points)
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=inputs.shape[0])
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(inputs)
        loss = -mll(output, targets)
        loss.backward()
        if (epoch+1) % (num_epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.3f}")
        optimizer.step()
    return model, likelihood

# Toy example
# TODO Build Data pipeline to feed into Inference Algorithm
# need to split our patient data 
if __name__ == "__main__":
    # More complex to data, N patients, with survival times T, two covariates uniformly disted
    N = 50
    T = torch.linspace(0.5, 5.0, N)
    X = torch.rand(N, 2)
    
    # Train the GP classifier using our data augmentation scheme
    model, likelihood = train_survival_gp(T, X, num_epochs=1_000)
    
    # Simplest Possible Inference - single data point
    # To predict the probability of an event (i.e. the acceptance probability \sigma(l(t,x)))
    # at a new time t_new and covariate x_new, we form an input and query the GP
    model.eval()
    likelihood.eval()
    t_new = torch.Tensor([3.0])
    x_new = torch.tensor([[0.5, 0.5]])
    new_input = torch.cat([t_new.unsqueeze(0), x_new], dim=1)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(new_input))
        # pred.mean is our estimate for σ(l(t,x)).
        print(f"Predicted acceptance probability (event) at t={t_new.flatten()}:", pred.mean.item())
    
        # To obtain a survival probability, one would typically combine this with the baseline hazard
        # For example, one may compute: S(t|x) \approx \exp(-\Gamma_0(t) * \sigma(l(t,x)))
        S_t = torch.exp(-Lambda0(t_new) * pred.mean)
        print("Estimated survival probability at t=3.0:", S_t.item())