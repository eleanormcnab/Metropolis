import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os


# Speed of light in km/s
c = 299792.458 

# Defining the class Cosmology
class Cosmology:
    """
    A class to model cosmology with methods to calculate the distance modulus.
    """

    def __init__(self, H0, Omega_m, Omega_lambda):
        """
        Initialising the Cosmology class and parameters 
        
        Parameters:
        ----------
        H0: float
            The Hubble constant in km/s/Mpc.
        Omega_m: float
            The matter density parameter.
        Omega_lambda: float
            The dark energy density parameter.
        """

        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.Omega_k = self.calculate_Omega_k()

    def integrand(self, z_prime):
        """
        Calculating the integrand of the distance D(z) function

        Parameters:
        ----------
        z_prime: float
            Redshift value.

        Returns:
        --------
        float
            Value of the integrand at redshift z_prime.
        """

        return 1 / ((self.Omega_m * (1 + z_prime) ** 3) + 
                    self.Omega_lambda + 
                    (self.Omega_k * (1 + z_prime) ** 2)) ** 0.5

    def calculate_Omega_k(self):
        """Calculating Omega_k"""
        return 1 - self.Omega_m - self.Omega_lambda

    def set_Omega_m(self, Omega_m):
        """Setting a new value for Omega_m"""

        self.Omega_m = Omega_m
        self.Omega_lambda = 1 - Omega_m - self.Omega_k

    def set_Omega_lambda(self, Omega_lambda):
        """Setting a new value for Omega_lambda"""

        self.Omega_lambda = Omega_lambda
        self.Omega_m = 1 - Omega_lambda - self.Omega_k

    def D_rectangle(self, z, n):
        """Using the rectangle rule to compute the distance D(z) with n intervals
        
        Parameters:
        ----------
        z: float
            Redshift value.
        
        n: int
            Number of intervals.
        
        Returns:
        --------
        float
            Distance D(z) calculated using the rectangle rule.
        """

        dz = z / n
        D_z = sum(self.integrand(i * dz) for i in range(n))
        return (c / self.H0) * dz * D_z

    def D_trapezoid(self, z, n):
        """Using the trapezoid rule to compute the distance D(z) with n intervals
        
        Parameters:
        ----------
        z: float
            Redshift value.
        
        n: int
            Number of intervals.
        
        Returns:
        --------
        float
            Distance D(z) calculated using the trapezoid rule.
        """

        dz = z / n
        D_z = 0.5 * (self.integrand(0) + self.integrand(z))
        D_z += sum(self.integrand(i * dz) for i in range(1, n))
        return (c / self.H0) * dz * D_z

    def D_simpson(self, z, n):
        """Using Simpson's rule to compute the distance D(z) with n intervals
        
        Parameters:
        ----------
        z: float
            Redshift value.
        
        n: int
            Number of intervals.
        
        Returns:
        --------
        float
            Distance D(z) calculated using Simpson's rule.
        """

        if n % 2 == 1:  
            n += 1
        dz = z / n
        D_z = self.integrand(0) + self.integrand(z)
        D_z += sum(4 * self.integrand(i * dz) for i in range(1, n, 2))
        D_z += sum(2 * self.integrand(i * dz) for i in range(2, n-1, 2))
        return (c / (3 * self.H0)) * dz * D_z

    def D_cumulative(self, z, n):
        """
        Using the cumulative form of the trapezoid rule to compute an array of D(z) values
        for z in the range [0, z] with n intervals.

        Parameters:
        ----------
        z: float
            Redshift value.
        
        n: int
            Number of intervals.
        
        Returns:
        --------
        array
            Array of z values.
        """

        dz = z / n
        z_values = [i * dz for i in range(n + 1)]
        y = [0]
        sum = 0
        for i in range(1, n + 1):
            sum += self.integrand((i - 0.5) * dz)
            y.append(sum * dz)
        D_z = [(c / self.H0) * val for val in y]
        return np.array(z_values), np.array(D_z)
       


    def interpolation(self, z, n):
        """
        Use the scipy interpolator to take an array of z values,
        which might be in any order, and return an array of corresponding distances D(z).

        Parameters:
        ----------
        z: array
            Array of redshift values.
        
        n: int
            Number of intervals.
        
        Returns:
        --------
        array
            Array of distances D(z).

        """

        z = np.array(z)
        # Use the cumulative distance function to calculate D(z) values
        z_grid, D_values = self.D_cumulative(z.max(), n)

        #Ensure that you don't extrapolate
        z_max = z_grid[-1]
        z = z[z <= z_max]
        # Interpolate the D(z) values with scipy interp1d
        interpolator = interp1d(z_grid, D_values)
        return interpolator(z)


    def distance_modulus(self, z, n):
        """
        Calculate the distance modulus for an array of redshift values z.

        Parameters:
        ----------
        z: array
            Array of redshift values.
        
        n: int
            Number of intervals.
        
        Returns:
        --------
        array
            Array of distance modulus values.  
        """
        
        # Interpolate distances for all redshift values
        D_z = self.interpolation(z, n)

        # Calculate the luminosity distance D_L
        # Flat universe:

        if self.Omega_k == 0:
            D_L = (1 + z) * D_z
        # Open universe
        elif self.Omega_k > 0:
            D_L = (1 + z) * (c / self.H0) * (1 / np.sqrt(self.Omega_k)) * np.sinh(np.sqrt(self.Omega_k) * (self.H0 / c) * D_z)
        # Closed universe
        else:
            D_L = (1 + z) * (c / self.H0) * (1 / np.sqrt(np.abs(self.Omega_k))) * np.sin(np.sqrt(np.abs(self.Omega_k)) * (self.H0 / c) * D_z)

        # Calculate the distance modulus
        distance_modulus = 5 * np.log10(D_L) + 25

        return distance_modulus
