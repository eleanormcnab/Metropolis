from cosmology import Cosmology as cosmo
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class Likelihood:
    def __init__(self, data_file):
        """
        Initialize the Likelihood class with cosmological parameters and observational data.

        Parameters:
        ----------
        cosmo : class
            An instance of the Cosmology class that calculates cosmological distances.
        data_file : str
            Path to the data file containing redshift, observed magnitude, and uncertainty.
        """
        # Load data and assign redshift, observed magnitudes, and uncertainties.
        self.data = np.loadtxt(data_file)
        self.z = self.data[:, 0]
        self.mobs = self.data[:, 1]
        self.sigma = self.data[:, 2]

    def model_magnitude(self, z, n, parameters):
        """
        Compute the predicted magnitude based on cosmological parameters.

        Parameters:
        ----------
        z : array
            Array of redshift values.
        n : int
            Number of intervals used for numerical integration.
        parameters : array
            Cosmological model parameters: [Omega_m, Omega_lambda, H0].

        Returns:
        -------
        array
            Predicted magnitudes for the given redshift values.
        """
        # Update cosmological parameters in the Cosmology instance.
        omega_m, omega_lambda, H0 = parameters
        c = cosmo(H0, omega_m, omega_lambda)
        # Use a fixed absolute magnitude of supernovae.
        M = -19.3
        return c.distance_modulus(z, n) + M


    def __call__(self, parameters, n, z=None, omega_lambda=0):
        """
        Call the log-likelihood function with optional model type.

        Parameters:
        ----------
        parameters : array
            Cosmological model parameters.
        n : int
            Number of intervals for numerical integration.
        z : array, optional
            Redshift values (uses data redshifts if not provided).
        omega_lambda : float, optional
            Value of Omega_lambda for the zero lambda model.

        Returns:
        -------
        float
            Log-likelihood of the model.
        """
       # Use the data redshifts if not provided.
        if z is None:
            z = self.z
        
    
    # Calculate the model magnitudes and return the log-likelihood.
        model_magnitude = self.model_magnitude(z, n, parameters)

        return - 0.5 * np.sum((self.mobs - model_magnitude) ** 2 / self.sigma ** 2)

    def convergence_test(self, N, parameters, tolerance):
        """
        Test the convergence of the likelihood calculation as the number of intervals increases.

        Parameters:
        ----------
        N : int
            Maximum number of intervals for integration.
        parameters : array
            Cosmological model parameters.
        tolerance : float
            Convergence limit for changes in likelihood.

        Returns:
        -------
        tuple
            Lists of likelihood values and their differences across intervals.
        """
        likelihoods = []
        delta_likelihoods = []

        # Avoid boundary issues by slightly adjusting redshifts.
        z_min = min(self.z)
        z_max = max(self.z)
        adjusted_z = np.array([min(max(z, z_min), z_max - 1e-10) for z in self.z])

        # Calculate likelihoods for increasing intervals and check for convergence.
        for n in range(1, N):
            log_likelihood = self.__call__(parameters, n, adjusted_z)
            likelihoods.append(log_likelihood)

            if len(likelihoods) > 1:
                delta_likelihood = abs(likelihoods[-1] - likelihoods[-2])
                delta_likelihoods.append(delta_likelihood)
                if delta_likelihood < tolerance:
                    print(f"Converged with {n} intervals")
                    break
        else:
            print("Convergence not reached")
        return likelihoods, delta_likelihoods

    def best_fit(self, initial_parameters, N, bounds, omega_lambda=0):
        """
        Find the best-fit cosmological parameters using maximum likelihood estimation.

        Parameters:
        ----------
        initial_parameters : array
            Initial guess for the parameters.
        N : int
            Number of intervals for numerical integration.
        bounds : list
            Bounds for the parameter values.

        Returns:
        -------
        tuple
            Best-fit parameters and maximum likelihood value.
        """
        # Minimize the negative log-likelihood function.
        result = minimize(lambda params: -self.__call__(params, N), initial_parameters, bounds=bounds)
        best_fit_parameters = result.x
        max_likelihood = -result.fun
        return best_fit_parameters, max_likelihood



def plot_convergence(likelihood_instance, initial_parameters, N, tolerance, omega_lambda=0):
    """
    Plot the convergence test for the likelihood function.

    Parameters:
    ----------
    likelihood_instance : class
        The Likelihood class instance.
    
    Plots: 
    -------
    Delta likelihood values against the number of intervals.
    """
    # Run the convergence test to obtain likelihood values and their differences
    likelihoods, delta_likelihoods = likelihood_instance.convergence_test(N, initial_parameters, tolerance)
    
    # Plot delta likelihoods, starting from the second interval
    plt.plot(range(2, len(likelihoods) + 1), delta_likelihoods)
    plt.axhline(y=tolerance, color='r', linestyle='--')  
    plt.text(2, tolerance * 1.1, 'Tolerance = 1e-4', color='r')
    plt.xlabel("Number of intervals")
    plt.ylabel("Delta likelihood")
    plt.title("Convergence test for the log likelihood function")
    plt.show()

def plot_best_fit(likelihood_instance, best_fit_parameters, N, omega_lambda=0):
    """
    Plot the observed data and the best-fit model.

    Parameters:
    ----------
    likelihood_instance : class
        The Likelihood class instance.
    
    Plots:
    -------
    Observed data and best-fit model.
    """
    # Extract data from the Likelihood instance
    z = likelihood_instance.z
    mobs = likelihood_instance.mobs
    sigma = likelihood_instance.sigma

    # Calculate best-fit model magnitudes
    best_fit_magnitude = likelihood_instance.model_magnitude(z, N, best_fit_parameters)
    
    # Plot observed data with error bars and the best-fit model
    plt.errorbar(z, mobs, yerr=sigma, fmt='x', label='Observed data')
    plt.plot(z, best_fit_magnitude, label='Best fit model')
    plt.xlabel("Redshift")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title("Observed data vs. Best-fit model")
    plt.show()

def residuals_plot(likelihood_instance, best_fit_parameters, N, omega_lambda=0):
    """
    Plot the residuals of the best-fit model.

    Parameters:
    ----------
    likelihood_instance : class
        The Likelihood class instance.
    
    Plots:
    -------
    Residuals of the best-fit model with error bars.

    Prints:
    -------
    Mean and standard deviation of the residuals.
    """
    # Extract data from the Likelihood instance
    z = likelihood_instance.z
    mobs = likelihood_instance.mobs
    sigma = likelihood_instance.sigma
    
    # Calculate residuals normalized by uncertainties
    best_fit_magnitude = likelihood_instance.model_magnitude(z, N, best_fit_parameters)
    residuals = (mobs - best_fit_magnitude) / sigma
    
    # Compute statistics for residuals
    mean = np.mean(residuals)
    std = np.std(residuals)
    print(f"Mean of residuals: {mean}")
    print(f"Standard deviation of residuals: {std}")
    
    # Plot residuals with a horizontal line at zero
    plt.errorbar(z, residuals, yerr=sigma, fmt='x')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Redshift")
    plt.ylabel("Residuals")
    plt.title("Residuals of Best-fit Model")
    plt.show()

def main():
    # Define initial parameters and load the data
    initial_parameters = [0.3, 0.7, 70]
    data_file = "pantheon_data.txt"
    cosmo_instance = cosmo(H0=70, Omega_m=0.3, Omega_lambda=0.7)
    likelihood_instance = Likelihood(data_file)
    # Set parameters for the convergence test and best-fit model
    N = 1000
    bounds = [(0, 1), (0, 1), (50, 100)]
    tolerance = 1e-4

    # Run convergence test and find best-fit parameters for the standard model
    plot_convergence(likelihood_instance, initial_parameters, N, tolerance)
    best_fit_parameters_standard, max_likelihood_standard = likelihood_instance.best_fit(initial_parameters, N, bounds)
    print(f"Best-fit parameters (standard model): {best_fit_parameters_standard}")
    print(f"Maximum likelihood (standard model): {max_likelihood_standard}")
    plot_best_fit(likelihood_instance, best_fit_parameters_standard, N)
    residuals_plot(likelihood_instance, best_fit_parameters_standard, N)

    # Run for the zero lambda modele
    likelihood_instance.__call__(initial_parameters, N, omega_lambda=0)
    best_fit_parameters_zero_lambda, max_likelihood_zero_lambda = likelihood_instance.best_fit(initial_parameters, N, bounds, omega_lambda=0)
    print(f"Best-fit parameters (zero lambda model): {best_fit_parameters_zero_lambda}")
    print(f"Maximum likelihood (zero lambda model): {max_likelihood_zero_lambda}")
    plot_best_fit(likelihood_instance, best_fit_parameters_zero_lambda, N, omega_lambda=0)
    residuals_plot(likelihood_instance, best_fit_parameters_zero_lambda, N, omega_lambda=0)



    
    
    
if __name__ == "__main__":
    main()