"""My personal take on Adverserial Variational Optimization."""


import math
import numpy as np
import random
import sys
import torch
import torch.nn.functional as F

from sklearn.utils import check_random_state
from torch.autograd import Variable


def main():
    # Assume there exists some true parameterization.
    # Beam Energy = 43 Gev, and Fermi's Constant is 0.9
    theta_true = [43.0, 0.9]
    # Assume there is an experiment drawing (real) samples from nature.
    p_r = real_experiment(theta_true, 100000)
    # Initialize the prior of theta, parameterized by a Gaussian.
    proposal = {'mu': [], 'sigma': []}
    # Check if a custom mu has been specified.
    if '--mu' in sys.argv:
        mu = sys.argv[sys.argv.index('--mu') + 1].split(",")
        mu = [float(e) for e in mu]
        proposal['mu'] = mu
        proposal['sigma'] = [.1, .1]
    else:
        # Add random beam energy.
        add_prior_beam_energy(proposal)
        # Add random Fermi constant.
        add_prior_fermi_constant(proposal)
    # Check if a custom sigma has been specified.
    if '--sigma' in sys.argv:
        sigma = sys.argv[sys.argv.index('--sigma') + 1].split(",")
        sigma = [float(e) for e in sigma]
        proposal['sigma'] = sigma
    else:
        # Initialize default sigma.
        proposal['sigma'] = [.1, .1]
    # Convert the proposal lists to PyTorch Tensors.
    proposal['mu'] = torch.FloatTensor(proposal['mu'])
    proposal['sigma'] = torch.FloatTensor(proposal['sigma'])
    # Inference on theta is done using a critic network in an adverserial setting.
    critic = CriticWithSigmoid(200)
    # Obtain the batch size from the arguments.
    if '--batch-size' in sys.argv:
        batch_size = int(sys.argv[sys.argv.index('--batch-size') + 1])
    else:
        batch_size = 256
    # Fit the proposal distribution to the real distribution using the critic.
    fit(proposal=proposal, p_r=p_r, critic=critic, theta_true=theta_true, batch_size=batch_size)
    # Display the current parameterization of the proposal distribution.
    print("\nProposal Distribution:")
    print(" - Beam Energy:")
    print("    mu: " + str(proposal['mu'][0]))
    print("    sigma: " + str(proposal['sigma'][0]))
    print(" - Fermi's Constant:")
    print("    mu: " + str(proposal['mu'][1]))
    print("    sigma: " + str(proposal['sigma'][1]))
    print("\nTrue Distribution:")
    print(" - Beam Energy: " + str(theta_true[0]))
    print(" - Fermi's Constant: " + str(theta_true[1]))


def fit(proposal, p_r, critic, theta_true, batch_size, num_iterations=1000):
    # Allocate the critic optimizer.
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.01)
    # Apply an initial fit of the critic parameters.
    fit_critic(proposal, p_r, critic, critic_optimizer, batch_size=batch_size, num_critic_iterations=1000)
    # Critic has an initial fit, apply variational proposal optimization.
    for iteration in range(0, num_iterations):
        print("True Mean: " + str(theta_true))
        print("Current Mean: " + str(proposal['mu']))
        print("Current Sigma: " + str(proposal['sigma']))
        # Fit the critic network (optimally.)
        fit_critic(proposal, p_r, critic, critic_optimizer, batch_size, 100)
        # Fit the proposal distribution.
        fit_proposal(proposal, critic, batch_size)


def fit_proposal(proposal, critic, batch_size):
    gradient_u_mu = torch.FloatTensor([0, 0])
    likelihoods = torch.zeros(batch_size)
    thetas = draw_gaussian(proposal, batch_size)
    # Compute the likelihood of every instance.
    for i in range(0, batch_size):
        x = Variable(simulator(thetas[i], 1))
        likelihoods[i] = critic(x).view(-1).data[0]
    # Fit a gradient.
    g, b = estimate_gradient(thetas, likelihoods)
    print("Slope for beam energy: " + str(g[0]))
    print("Slope for fermi constant: " + str(g[1]))
    proposal['mu'] -= g


def fit_critic(proposal, p_r, critic, optimizer, batch_size, num_critic_iterations=1000):
    loss_function = torch.nn.MSELoss()
    zeros = Variable(torch.zeros(batch_size)) # Real
    ones = Variable(torch.ones(batch_size)) # Fake
    # Fit the critic optimally.
    for iteration in range(0, num_critic_iterations):
        # Generate the simulation data.
        x_g = sample_generated_data(proposal, batch_size)
        # Fetch the real data.
        x_r = sample_real_data(p_r, batch_size)
        # Reset the gradients.
        critic.zero_grad()
        # Forward pass with real data.
        y_predicted_r = critic(x_r)
        loss_real = loss_function(y_predicted_r, zeros)
        # Forward pass with generated data.
        y_predicted_g = critic(x_g)
        loss_fake = loss_function(y_predicted_g, ones)
        # Compute the loss, and the accompanying gradients.
        loss = (loss_real + loss_fake)
        loss.mean().backward()
        optimizer.step()
        if iteration % 100 == 0:
            print("At iteration " + str(iteration) + " - Loss: " + str(loss.mean().data.numpy()[0]))
    # Display the loss of the critic at the last step.
    print("Loss: " + str(loss.mean().data.numpy()[0]))


def estimate_gradient(thetas, likelihoods):
    model = torch.nn.Linear(thetas.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = Variable(thetas, requires_grad=True)
    y = Variable(10 * likelihoods)
    for iteration in range(0, 10000):
        model.zero_grad()
        loss = F.smooth_l1_loss(model(x), y)
        loss.backward()
        optimizer.step()
    model_parameters = [x.data for x in list(model.parameters())]
    slope = model_parameters[0]
    bias = model_parameters[1]

    return slope.view(-1), bias.view(-1)


def add_prior_beam_energy(prior):
    g = random_gaussian(mu=[40, 50], sigma=1.0)
    add_prior(prior, g['mu'], g['sigma'])


def add_prior_fermi_constant(prior):
    g = random_gaussian(mu=[0, 2], sigma=1.0)
    add_prior(prior, g['mu'], g['sigma'])


def add_prior(prior, mu, sigma):
    prior['mu'].append(mu)
    prior['sigma'].append(sigma)


def random_gaussian(mu=[-1, 1], sigma=5.0):
    return {'mu': np.random.uniform(mu[0], mu[1]),
            'sigma': np.random.uniform(0.0, sigma)}


def draw_gaussian(d, num_samples, random_state=None):
    num_parameters = len(d['mu'])
    mu = d['mu']
    sigma = d['sigma']
    thetas = torch.zeros((num_samples, num_parameters))
    for i in range(0, num_samples):
        gaussian = torch.normal(mu, sigma)
        thetas[i, :] = gaussian

    return thetas


def real_experiment(theta, n_samples):
    return simulator(theta, n_samples)


def simulator(theta, n_samples, random_state=None):
    rng = check_random_state(random_state)
    samples = simulator_rej_sample_costheta(n_samples, theta, rng)

    return torch.from_numpy(samples.reshape(-1, 1)).float()


def simulator_rej_sample_costheta(n_samples, theta, rng):
    sqrtshalf = theta[0]
    gf = theta[1]

    ntrials = 0
    samples = []
    x = torch.linspace(-1, 1, steps=1000)
    maxval = torch.max(simulator_diffxsec(x, sqrtshalf, gf))

    while len(samples) < n_samples:
        ntrials = ntrials + 1
        xprop = rng.uniform(-1, 1)
        ycut = rng.rand()
        yprop = (simulator_diffxsec(xprop, sqrtshalf, gf) / maxval)[0]
        if (yprop / maxval) < ycut:
            continue
        samples.append(xprop)

    return np.array(samples)


def simulator_diffxsec(costheta, sqrtshalf, gf):
    norm = 2. * (1. + 1. / 3.)
    return ((1 + costheta ** 2) + simulator_a_fb(sqrtshalf, gf) * costheta) / norm


def simulator_a_fb(sqrtshalf, gf):
    mz = 90
    gf_nom = 0.9
    sqrts = sqrtshalf * 2.
    x = torch.FloatTensor([(sqrts - mz) / mz * 10])
    a_fb_en = torch.tanh(x)
    a_fb_gf = gf / gf_nom

    return 2 * a_fb_en * a_fb_gf


def sample_real_data(p_r, batch_size=256):
    samples = torch.zeros((batch_size, 1))
    num_samples_p_r = len(p_r)
    for index in range(0, batch_size):
        random_index = random.randint(0, num_samples_p_r - 1)
        samples[index, :] = p_r[random_index]

    return torch.autograd.Variable(samples, requires_grad=True)


def sample_generated_data(proposal, batch_size=256):
    # Sample `batch_size` thetas according to our proposal distribution.
    thetas = draw_gaussian(proposal, batch_size)
    # Obtain the individual Gaussians.
    theta_beam_energy = thetas[:, 0]
    theta_fermi_constant = thetas[:, 1]
    # Sample according to the proposal distribution.
    samples = torch.zeros((batch_size, 1))
    for sample_index, theta in enumerate(thetas):
        samples[sample_index, :] = simulator(theta, 1)

    return torch.autograd.Variable(samples, requires_grad=True)


class CriticWithSigmoid(torch.nn.Module):

    def __init__(self, num_hidden):
        super(CriticWithSigmoid, self).__init__()
        self.fc_1 = torch.nn.Linear(1, num_hidden)
        self.fc_2 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc_3 = torch.nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.sigmoid(self.fc_3(x))

        return x


if __name__ == '__main__':
    main()
