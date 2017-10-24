
# Adverserial Variational Optimization

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
    # Beam Energy = 45 Gev, and Fermi's Constant is 0.9
    theta_true = [45.0, 0.9]
    # Assume there is an experiment drawing (real) samples from nature.
    p_r = real_experiment(theta_true, 10000)
    # Initialize the prior of theta, parameterized by a Gaussian.
    proposal = {'mu': [], 'sigma': []}
    # Check if a custom mu has been specified.
    if '--mu' in sys.argv:
        mu = sys.argv[sys.argv.index('--mu') + 1].split(",")
        mu = [float(e) for e in mu]
        proposal['mu'] = mu
        proposal['sigma'] = [1., 1.]
    else:
        # Add random beam energy.
        add_prior_beam_energy(proposal)
        # Add random Fermi constant.
        add_prior_fermi_constant(proposal)
    # Check if a custom sigma has been specified.
    if '--sigma' in sys.argv and '--mu' in sys.argv:
        sigma = sys.argv[sys.argv.index('--sigma') + 1].split(",")
        sigma = [float(e) for e in sigma]
        proposal['sigma'] = sigma
    # Convert the proposal lists to PyTorch Tensors.
    proposal['mu'] = torch.FloatTensor(proposal['mu'])
    proposal['sigma'] = torch.FloatTensor(proposal['sigma'])
    # Inference on theta is done using a critic network in an adverserial setting.
    if '--sigmoid' in sys.argv:
        critic = CriticWithSigmoid(num_hidden=50)
    else:
        critic = Critic(num_hidden=50)
    # Obtain the batch size from the arguments.
    if '--batch-size' in sys.argv:
        batch_size = int(sys.argv[sys.argv.index('--batch-size') + 1])
    else:
        batch_size = 256
    # Fit the proposal distribution to the real distribution using the critic.
    fit(proposal, p_r, critic, theta_true, batch_size)
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


def fit(proposal, p_r, critic, theta_true, num_iterations=1000, batch_size=256):
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)
    for iteration in range(0, num_iterations):
        print("True Mu: " + str(theta_true))
        print("Current Mu: " + str(proposal['mu']))
        print("Current Sigma: " + str(proposal['sigma']))
        # Fit the critic network.
        fit_critic(proposal, p_r, critic, critic_optimizer, batch_size=batch_size, num_critic_iterations=1000)
        # Fit the proposal distribution.
        fit_proposal(proposal, p_r, critic, batch_size)


def fit_critic(proposal, p_r, critic, optimizer, num_critic_iterations=50000, batch_size=256):
    # Fetch the data batches.
    x_r = sample_real_data(p_r, batch_size)
    x_g = sample_generated_data(proposal, batch_size)
    # Fit the critic optimally.
    for iteration in range(0, num_critic_iterations):
        # Reset the gradients.
        critic.zero_grad()
        # Forward pass with real data.
        y_r = critic(x_r).mean()
        # Forward pass with generated data.
        y_g = critic(x_g).mean()
        # Obtain gradient penalty (GP).
        gp = compute_gradient_penalty(critic, x_r.data, x_g.data).mean()
        # Compute the loss, and the accompanying gradients.
        loss = y_g - y_r + gp
        loss.backward()
        optimizer.step()
        # Check if debugging information needs to be shown.
        if iteration % 1000 == 0:
            mean_loss = loss.data.numpy()[0]
            print("Loss: " + str(mean_loss))


def fit_proposal(proposal, p_r, critic, batch_size=256, gamma=5.0):
    gradient_u_mu = torch.FloatTensor([0, 0])
    gradient_u_sigma = torch.FloatTensor([0, 0])
    gradient_entropy_sigma = torch.FloatTensor([0, 0])
    # Draw several thetas from the current proposal distribution.
    thetas = draw_gaussian(proposal, batch_size)
    # Compute the q-gradient for every theta.
    for theta in thetas:
        # Draw a sample from the simulator.
        x = torch.autograd.Variable(simulator(theta, 1))
        likelihood_x = critic(x).view(-1)
        mu = torch.autograd.Variable(proposal['mu'], requires_grad=True)
        sigma = torch.autograd.Variable(proposal['sigma'], requires_grad=True)
        # Compute the gradient of the Gaussian logpdf.
        theta = torch.autograd.Variable(theta)
        logpdf = gaussian_logpdf(mu, sigma, theta)
        logpdf.sum().backward()
        gradient_logpdf_mu = mu.grad.data
        gradient_logpdf_sigma = sigma.grad.data
        # Add the logpdf gradient to the current variational upperbound.
        gradient_u_mu += -likelihood_x.data * gradient_logpdf_mu
        gradient_u_sigma += -likelihood_x.data * gradient_logpdf_sigma
        # Clean the gradients of the variable.
        mu.grad.data.zero_()
        sigma.grad.data.zero_()
    # Compute the gradient of the entropy.
    sigma = torch.autograd.Variable(proposal['sigma'], requires_grad=True)
    differential_entropy = gaussian_differential_entropy(sigma)
    differential_entropy.backward()
    gradient_entropy_sigma = -sigma.grad.data
    # Compute the final adverserial gradient.
    gradient_u_mu = 0.1 * ((1. / batch_size) * gradient_u_mu)
    gradient_u_sigma = 0.1 * ((1. / batch_size) * gradient_u_sigma + gamma * gradient_entropy_sigma)
    print(gradient_u_sigma)
    # Apply the gradient to the proposal distribution.
    proposal['mu'] += gradient_u_mu
    proposal['sigma'] -= gradient_u_sigma


def compute_gradient_penalty(critic, real, fake, l=5.0):
    # Compute x_hat and its output.
    epsilon = torch.rand(real.size())
    x_hat = epsilon * real + ((1. - epsilon) * fake)
    x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
    y_hat = critic(x_hat)
    # Compute the associated gradients.
    gradients = torch.autograd.grad(outputs=y_hat, inputs=x_hat,
                                    grad_outputs=torch.ones(y_hat.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # Prevent norm 0 causing NaN.
    gradients = gradients + 1e-16
    # Compute the gradient penalty.
    gradient_penalty = l * ((gradients.norm(2, dim=1) - 1.) ** 2)

    return gradient_penalty


def sample_real_data(p_r, batch_size=256):
    samples = torch.zeros((batch_size, 1))
    num_samples_p_r = len(p_r)
    for index in range(0, batch_size):
        random_index = random.randint(0, num_samples_p_r - 1)
        samples[index, :] = p_r[random_index]

    return torch.autograd.Variable(samples, requires_grad=False)


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

    return torch.autograd.Variable(samples, requires_grad=False)


def gaussian_logpdf(mu, sigma, theta):
    # Define `a` as np.log((2. * np.pi) ** 0.5)
    a = 0.91893853320467267
    # Compute the logpdf.
    logpdf = -(sigma.log() + a + (theta - mu) ** 2 / (2. * sigma ** 2))

    return logpdf


def gaussian_differential_entropy(sigma):
    dentropy = -0.5 * (2. * np.pi * np.e * sigma ** 2).log()

    return dentropy.sum()


def add_prior_beam_energy(prior):
    g = random_gaussian(mu=[30, 60], sigma=1.0)
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
    thetas = torch.zeros((num_samples, num_parameters))
    mu = d['mu']
    sigma = d['sigma']
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


class Critic(torch.nn.Module):

    def __init__(self, num_hidden):
        super(Critic, self).__init__()
        self.fc_1 = torch.nn.Linear(1, num_hidden)
        self.fc_2 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc_3 = torch.nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = (self.fc_3(x))

        return x


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
