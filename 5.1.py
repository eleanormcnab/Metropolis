from unit4 import Likelihood as L
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from cosmology import Cosmology as cosmo
import time as time

def likelihood_grid(n, omega_m_range, omega_lambda_range, H0_range):
    """
    Making a 3D likelihood grid of the cosmological parameters Omega_m, Omega_lambda and H0.
    Parameters:
    -----------
    n: int
        Number of sample points in each dimension.
    omega_m_range: tuple
        Range of Omega_m values (min, max).
    omega_lambda_range: tuple
        Range of Omega_lambda values (min, max).
    H0_range: tuple
        Range of H0 values (min, max).

    Returns:
    --------
    grid: array
        3D array of the likelihood values for each combination of Omega_m, Omega_lambda and H0.
    """

    # Setting the range of values for each parameter
    omega_m_values = np.linspace(omega_m_range[0], omega_m_range[1], n)
    omega_lambda_values = np.linspace(omega_lambda_range[0], omega_lambda_range[1], n)
    H0_values = np.linspace(H0_range[0], H0_range[1], n)

    # Creating a 3D grid with dimensions of the number of values for each parameter
    grid = np.zeros((n, n, n))

    # Calculating the likelihood using the Likelihood class
    likelihood = L('pantheon_data.txt')

    # Looping over each combination of parameters
    for i, omega_m in enumerate(omega_m_values):
        for j, omega_lambda in enumerate(omega_lambda_values):
            for k, H0 in enumerate(H0_values):
                # Create an instance of the Cosmology class
                cosmology = cosmo(omega_m, omega_lambda, H0)
                # Set parameters for the Likelihood class
                parameters = [omega_m_values[i], omega_lambda_values[j], H0_values[k]]

                # Store the likelihood value in the grid
                grid[i, j, k] = likelihood(parameters, n)
    return grid

def marginalise_2D(grid):
    """ 
    Marginalising the 3D likelihood grid over Omega_m, Omega_lambda and H0 to get 2D marginalised grids.
    Parameters:
    -----------
    grid: array
        3D array of the likelihood values for each combination of Omega_m, Omega_lambda and H0.

    Returns:
    --------
    marginalised_omega_lambda_H0: array
        2D array of the marginalised likelihood values for Omega_lambda and H0.
    marginalised_omega_m_H0: array
        2D array of the marginalised likelihood values for Omega_m and H0.
    marginalised_omega_m_omega_lambda: array
        2D array of the marginalised likelihood values for Omega_m and Omega_lambda.
    """

    # Using the exponential of the grid, remembering we are working with log likelihoods
    max_value = np.max(grid)
    grid_exp = np.exp(grid - max_value)

    # Marginalising over each parameter by summing over the axes
    marginalised_omega_lambda_H0 = np.sum(grid_exp, axis=0)
    marginalised_omega_m_H0 = np.sum(grid_exp, axis=1)
    marginalised_omega_m_omega_lambda = np.sum(grid_exp, axis=2)

    return marginalised_omega_lambda_H0, marginalised_omega_m_H0, marginalised_omega_m_omega_lambda

def marginalise_1D(grid):
    """
    Marginalising the 3D likelihood grid over Omega_m, Omega_lambda and H0 to get 1D marginalised likelihoods.
    Parameters:
    -----------
    grid: array
        3D array of the likelihood values for each combination of Omega_m, Omega_lambda and H0.

    Returns:
    --------
    marginalised_H0: array
        1D array of the marginalised likelihood values for H0.
    marginalised_omega_lambda: array
        1D array of the marginalised likelihood values for Omega_lambda.
    marginalised_omega_m: array
        1D array of the marginalised likelihood values for Omega_m.
    """

    #Again, using the exponential of the grid
    max_value = np.max(grid)
    grid_exp = np.exp(grid - max_value)

    #Marginalising over each parameter by summing over the other two parameter axes
    marginalised_H0 = np.sum(grid_exp, axis=(0, 1))
    marginalised_omega_lambda = np.sum(grid_exp, axis=(0, 2))
    marginalised_omega_m = np.sum(grid_exp, axis=(1, 2))

    return marginalised_H0, marginalised_omega_lambda, marginalised_omega_m

def main():
    # Set number of sample points in each dimension
    n = 100

    # Define parameter ranges
    omega_m_range = (0, 0.5)
    omega_lambda_range = (0.1, 0.9)
    H0_range = (69, 72)

    # Start the timer
    start = time.time()

    # Create the 3D likelihood grid
    grid = likelihood_grid(n, omega_m_range, omega_lambda_range, H0_range)
    
    # End the timer
    end = time.time()
    print("Time taken to create the grid: ", end - start)

    # Marginalise the 3D grid to get 2D marginalised grids
    marginalised_omega_lambda_H0, marginalised_omega_m_H0, marginalised_omega_m_omega_lambda = marginalise_2D(grid)

    # Marginalise the 3D grid to get 1D marginalised likelihoods
    marginalised_H0, marginalised_omega_lambda, marginalised_omega_m = marginalise_1D(grid)

    # Plotting the 2D marginalised grids
    plt.subplot(1, 3, 1)
    plt.imshow(marginalised_omega_lambda_H0.T, origin='lower', aspect='auto', extent=[H0_range[0], H0_range[1], omega_lambda_range[0], omega_lambda_range[1]])
    plt.colorbar(label='Likelihood')
    plt.xlabel('H0')
    plt.ylabel('Omega_lambda')
    plt.title('Omega_lambda vs H0')

    plt.subplot(1, 3, 2)
    plt.imshow(marginalised_omega_m_H0.T, origin='lower', aspect='auto', extent=[H0_range[0], H0_range[1], omega_m_range[0], omega_m_range[1]])
    plt.colorbar(label='Likelihood')
    plt.xlabel('H0')
    plt.ylabel('Omega_m')
    plt.title('Omega_m vs H0')

    plt.subplot(1, 3, 3)
    plt.imshow(marginalised_omega_m_omega_lambda.T, origin='lower', aspect='auto', extent=[omega_lambda_range[0], omega_lambda_range[1], omega_m_range[0], omega_m_range[1]])
    plt.colorbar(label='Likelihood')
    plt.xlabel('Omega_m')
    plt.ylabel('Omega_lambda')
    plt.title('Omega_m vs Omega_lambda')

    plt.show()

    # Plotting the 1D marginalised likelihoods
    plt.figure()
    plt.plot(np.linspace(H0_range[0], H0_range[1], n), marginalised_H0)
    plt.xlabel('H0')
    plt.ylabel('Likelihood')
    plt.title('Marginalised likelihood for H0')
    plt.show()

    plt.figure()
    plt.plot(np.linspace(omega_lambda_range[0], omega_lambda_range[1], n), marginalised_omega_lambda)
    plt.xlabel('Omega_lambda')
    plt.ylabel('Likelihood')
    plt.title('Marginalised likelihood for Omega_lambda')
    plt.show()

    plt.figure()
    plt.plot(np.linspace(omega_m_range[0], omega_m_range[1], n), marginalised_omega_m)
    plt.xlabel('Omega_m')
    plt.ylabel('Likelihood')
    plt.title('Marginalised likelihood for Omega_m')
    plt.show()

if __name__ == '__main__':
    main()

