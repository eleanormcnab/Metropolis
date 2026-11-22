import numpy as np
from unit4 import Likelihood as L
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from cosmology import Cosmology as cosmo
import time as time

class Metropolis:
    def __init__(self, initial_parameters, n_steps, step_sizes, n_burn_in, n_integral):
        """
        Initialise the Metropolis class with likelihood function, initial parameters, number of steps, step sizes, number of burn-in steps and number of integral steps.

        Parameters:
        ----------
        initial_parameters : array
            An initial guess for the three cosmological parameters.
        n_steps : int
            The number of steps in the Metropolis algorithm.
        step_sizes : array
            The step sizes for each parameter.
        n_burn_in : int
            The number of burn-in steps.
        n_integral : int
            The number of steps for the integral.
        """
        # Store the initial parameters, number of steps, step sizes, number of burn-in steps and number of integral steps.
        self.initial_parameters = np.array(initial_parameters)
        self.n_steps = n_steps
        self.step_sizes = np.array(step_sizes)
        self.n_burn_in = n_burn_in
        self.n_integral = n_integral

    def proposal(self, parameters):
        """
        Propose new parameters based on the current parameters and step sizes.

        Parameters:
        ----------
        parameters : array
            The current parameters.

        Returns:
        --------
        array
            The proposed parameters.
        """
        # Generate new parameters by adding random numbers to the current parameters.
        # The random numbers are drawn from a normal distribution with mean 0 and standard deviation 1.
        # They are then multiplied by the step sizes and added to the current parameters.
        new_parameters = np.random.normal(0, 1, size=len(parameters)) * self.step_sizes + parameters
        return new_parameters

    def start_chain(self, likelihood):
        """
        Start the chain.

        Parameters:
        ----------
        likelihood : function
            The likelihood function.

        Returns:
        --------
        tuple
            The chain, log likelihoods, and initial log likelihood.
        """
        # Create an empty chain and log likelihoods list.
        chain = []
        log_likelihoods = []

        # Calculate log likelihood and append it to the log likelihoods list, and append the initial parameters to the chain.
        log_likelihood_initial = likelihood(self.initial_parameters, self.n_integral)
        log_likelihoods.append(log_likelihood_initial)
        chain.append(self.initial_parameters)
        return chain, log_likelihoods, log_likelihood_initial

    def metropolis_step(self, likelihood, current_parameters, log_likelihood_current):
        """
        Perform a single Metropolis step.

        Parameters:
        ----------
        likelihood : function
            The likelihood function.
        current_parameters : array
            The current parameters.
        log_likelihood_current : float
            The log likelihood of the current parameters.

        Returns:
        --------
        tuple
            The new parameters and their log likelihood.
        """
        # Propose new parameters and calculate the log likelihood.
        new_parameters = self.proposal(current_parameters)
        log_likelihood_new = likelihood(new_parameters, self.n_integral)

        # If the new log likelihood is greater than the current log likelihood, accept the new parameters.
        if log_likelihood_new > log_likelihood_current:
            return new_parameters, log_likelihood_new
        # If the new log likelihood is less than the current log likelihood, compare the ratio of the new log likelihood to the current log likelihood with a random number drawn from a uniform distribution.
        else:
            u = np.random.uniform(0, 1)
            # If the ratio is greater than the random number, accept the new parameters.
            if np.log(u) < log_likelihood_new - log_likelihood_current:
                return new_parameters, log_likelihood_new
            # Otherwise, keep the current parameters.
            else:
                return current_parameters, log_likelihood_current

    def run_metropolis(self, likelihood):
        """
        Run the Metropolis algorithm to generate a chain of parameters.

        Parameters:
        ----------
        likelihood : function
            The likelihood function.

        Returns:
        --------
        list
            The chain of parameters.
        """
        # Start the chain and log likelihoods.
        chain, log_likelihoods, log_likelihood_current = self.start_chain(likelihood)
        # Perform the Metropolis steps over the specified number of steps.
        for step in range(1, self.n_steps):
            if step % 100 == 0:
                print(f"Step {step}")
            # Perform a single Metropolis step, which will then be appended to the chain.
            current_parameters, log_likelihood_current = self.metropolis_step(
                likelihood, chain[-1], log_likelihood_current
            )
            log_likelihoods.append(log_likelihood_current)
            chain.append(current_parameters)

        # Discard the burn-in steps, where the chain is still converging to the target distribution.
        chain = chain[self.n_burn_in:]
        log_likelihoods = log_likelihoods[self.n_burn_in:]
        return chain

    def __call__(self):
        """
        Run the Metropolis algorithm and return the chain of parameters.

        Returns:
        --------
        list
            The chain of parameters.
        """
        likelihood = L('pantheon_data.txt')
        return self.run_metropolis(likelihood)

def plot_1D_results(chain):
    """
    Plot 1D histograms of each parameter in the chain.

    Parameters:
    ----------
    chain : list
        The chain of parameters.
    """
    # Convert the chain to a numpy array
    chain = np.array(chain)
    # Plot 1D histograms of the parameters
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(chain[:, 0], bins=50, color='b', alpha=0.7)
    axes[0].set_xlabel(r'$\Omega_m$')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(chain[:, 1], bins=50, color='r', alpha=0.7)
    axes[1].set_xlabel(r'$\Omega_{\Lambda}$')
    axes[1].set_ylabel('Frequency')

    axes[2].hist(chain[:, 2], bins=50, color='g', alpha=0.7)
    axes[2].set_xlabel(r'$H_0$')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_2D_results(chain):
    """
    Plot 2D histograms of each pair of parameters in the chain.

    Parameters:
    ----------
    chain : list
        The chain of parameters.
    """
    # Convert the chain to a numpy array
    chain = np.array(chain)
    # Plot 2D histograms of the parameters
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist2d(chain[:, 0], chain[:, 1], bins=50, cmap='Blues')
    axes[0].set_xlabel(r'$\Omega_m$')
    axes[0].set_ylabel(r'$\Omega_{\Lambda}$')

    axes[1].hist2d(chain[:, 0], chain[:, 2], bins=50, cmap='Reds')
    axes[1].set_xlabel(r'$\Omega_m$')
    axes[1].set_ylabel(r'$H_0$')

    axes[2].hist2d(chain[:, 1], chain[:, 2], bins=50, cmap='Greens')
    axes[2].set_xlabel(r'$\Omega_{\Lambda}$')
    axes[2].set_ylabel(r'$H_0$')

    plt.tight_layout()
    plt.show()

def plot_3D_results(chain):
    """
    Plot 3D results, by creating 2D scatter plots of each pair of parameters in the chain, with the third parameter as an extra dimension using colour.

    Parameters:
    ----------
    chain : list
        The chain of parameters.
    """
    # Convert the chain to a numpy array
    chain = np.array(chain)
    # Plot 3D histograms of the parameters
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Omega_m vs Omega_lambda on a scatter plot
    sc = axes[0].scatter(chain[:, 0], chain[:, 1], s=1, c=chain[:, 2], cmap='viridis')
    axes[0].set_xlabel(r'$\Omega_m$')
    axes[0].set_ylabel(r'$\Omega_{\Lambda}$')
    axes[0].set_title(r'$\Omega_m$ vs $\Omega_{\Lambda}$')
    cbar = plt.colorbar(sc, ax=axes[0])
    cbar.set_label(r'$H_0$')

    # Plot Omega_m vs H0 on a scatter plot
    sc = axes[1].scatter(chain[:, 0], chain[:, 2], s=1, c=chain[:, 1], cmap='viridis')
    axes[1].set_xlabel(r'$\Omega_m$')
    axes[1].set_ylabel(r'$H_0$')
    axes[1].set_title(r'$\Omega_m$ vs $H_0$')
    cbar = plt.colorbar(sc, ax=axes[1])
    cbar.set_label(r'$\Omega_{\Lambda}$')

    # Plot Omega_lambda vs H0 on a scatter plot
    sc = axes[2].scatter(chain[:, 1], chain[:, 2], s=1, c=chain[:, 0], cmap='viridis')
    axes[2].set_xlabel(r'$\Omega_{\Lambda}$')
    axes[2].set_ylabel(r'$H_0$')
    axes[2].set_title(r'$\Omega_{\Lambda}$ vs $H_0$')
    cbar = plt.colorbar(sc, ax=axes[2])
    cbar.set_label(r'$\Omega_m$')

    plt.tight_layout()
    plt.show()

def main():
    # Set the initial parameters, number of steps, step sizes, number of burn-in steps and number of integral steps.
    initial_parameters = [0.3, 0.6, 70]
    n_steps = 10000
    n_integral = 250
    step_sizes = [0.07, 0.1, 0.3]
    n_burn_in = 1000

    # Start the timer
    start = time.time()

    # Create an instance of the Metropolis class and run the Metropolis algorithm
    m = Metropolis(initial_parameters, n_steps, step_sizes, n_burn_in, n_integral)
    chain = m()

    # Plot 1D and 2D and 3D histograms of the results
    plot_1D_results(chain)
    plot_2D_results(chain)
    plot_3D_results(chain)

    # End the timer
    end = time.time()
    print("Time taken to run Metropolis algorithm: ", end - start)

if __name__ == "__main__":
    main()