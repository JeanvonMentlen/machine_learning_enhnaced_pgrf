from scipy.special import hermite, erf, erfinv
from scipy.optimize import minimize, fsolve
from scipy.integrate import quad, dblquad
import json
from math import factorial
from numpy import sin, cos, cosh, sinh, tan, zeros, pi, sqrt, power, exp, array, linspace, abs, matmul, outer, sum, empty, vectorize
import numpy as np
from optimparallel import minimize_parallel
# A = Li2S4, B = Li2S, S = EL

"""
plurigaussianrandomfield.py

This module contains the PluriGaussianRandomField class, which is designed to simulate three phased hierarchical 
systems. 

The PluriGaussianRandomField class initializes a plethora of parameters within its constructor (__init__ method) 
to optimize computational speed during subsequent operations. This design choice prioritizes performance, particularly 
for calculating many intensity curves for different PGRF parameters, but the same q_vector. One iteration of 
calculate_PGRF_intensity takes roughly 2 seconds on a reference system.

Usage Example:
--------------

q_vector = np.load("path_to_data")
params = dict(parameter values)
field = PluriGaussianRandomField(q_vector, [q_min, q_max])
result = field.calculate_PGRF_intensity(int_const=params['int_const'], phi_A=params['phi_A'],
                                              vol_factor=params['vol_factor'], beta_angle=params['beta_angle'],
                                              l_z=params['l_z'], l_y=params['l_y'], b=params['b'],
                                              porod_const=params['porod_const'], d_y=params['d_y'],
                                              d_z=params['d_z'])

Attributes:
-----------
- q_vector : numpy array containing the q values
-  q_range : list containing q_min and q_max

Methods:
--------
- calculate_PGRF_intensity: 
    Input:
        - int_const (float): shifts the intensity curve up or down
        - phi_A (float): volume fraction of phase A. Tested range: [0, 0.33]
        - volume_factor (float): ratio between volume fraction of phase S and A, Tested range: [0, 2], possible to
                                 replace with phi_S as parameter
        - beta_angle (float): correlation between phase A and S. Expected range: [30, 90]. Integral error large if <30
        - l_z (float): peak position of phase A (z-field), Tested range: [0.1, 3]
        - l_y (float): peak position of phase B (y-field), Tested range: [3, 10]
        - b (float): correlation parameter between phase A and S. Tested range: [0.1, 10]
        - porod_const (float): prefactor for porod decay at low q. Tested range: [0, 1]
        - d_y (float): peak shape factor for phase A (z-field): Tested range: [1, 10] 
        - d_y (float): peak shape factor for phase B (y-field): Tested range: [10, 150] 
    
    Output: 
        - Scattering intensity as numpy array
"""

class PluriGaussianRandomField:
    def __init__(self, q_vector, q_range):
        """
        Constructor for the PluriGaussianRandomField class.

        Parameters:
        -----------
        q_vector : numpy array
           nd array containing the q values at which the intensity will be evaluated. Given, e.g., by experimental
           data. Should be in 1/nm
        q_range : list
           range selection from the q_vector
        ...
        """
        self.data = {'q': q_vector}

        self.qmin_num = int(q_range[0])  # 12
        self.qmax_num = int(q_range[1])  # 439

        # load material and simulation params from json file
        with open('../configs/pgrf_params.json') as data:
            self.params = json.load(data)

        # integration and summation range
        self.m, self.gaussmin, self.gaussnum, self.pmax, self.nmax,  = 320, -8, 16, 40, 40
        self.num_q, self.num_r = len(self.data['q']), self.params['num_r']
        self.dy, self.dz = self.gaussnum / self.m, self.gaussnum / self.m
        self.q_space = linspace(0, self.num_q - 1, self.num_q, dtype=int)
        self.m_space = linspace(0, self.m - 1, self.m, dtype=int)
        self.r_space = linspace(0, self.num_r - 1, self.num_r, dtype=int)
        self.p_space = linspace(0, self.pmax - 1, self.pmax, dtype=int)
        self.n_space = linspace(0, self.nmax - 1, self.nmax, dtype=int)
        self.r = array([self.params['r_max'] / self.num_r + ii * self.params['r_max'] / self.num_r for ii in
                        linspace(0, self.num_r - 1, self.num_r)], dtype=np.float128)
        self.coordinate = [self.gaussmin + self.gaussnum * ii / self.m for ii in self.m_space]

        self.surface_area = {
            'S': 0,
            'A': 0,
            'B': 0
        }

        self.results = {
            'Intensity': empty(self.num_q),
            'two_point_probability': None,
            'phase': dict()
        }

        self.generate_hermite_integral_matrix()

    def load_experimental_data(self, data):
        # load from dataframe; columns named according to ILL data structure
        # q is in Ångström
        data_dict = {
            'q': data[:, 0],
            'Intensity': data[:, 1]
        }
        return data_dict

    # main function; to be optimized;
    def calculate_PGRF_intensity(self, int_const, phi_A, vol_factor, l_z, l_y, beta_angle, b, porod_const, d_y=None, d_z=None, rho_A=None, rho_B=None, rho_S=None):
        #ensure that all paramaters are 0 at the beginning
        # convert beta directly to radians
        beta = beta_angle * pi / 180
        phi_S = phi_A * vol_factor
        phi_B = 1 - phi_S - phi_A

        if d_y is None:
            d_y = self.params['d_y']
        if d_z is None:
            d_z = self.params['d_z']

        # SLD are normally set in the pgrf_params.json file.
        # To create trainings data with varying SLD, they can be given as input here
        # if rho_A is not None:
        #     self.params['rho_A'] = rho_A
        # if rho_B is not None:
        #     self.params['rho_B'] = rho_B
        # if rho_S is not None:
        #     self.params['rho_S'] = rho_S

        # GRF values
        two_point_probability = {
            'SS': zeros(self.num_r),
            'AA': zeros(self.num_r),
            'BB': zeros(self.num_r)
        }
        # self.phi_S = self.phi_A**2 #in certain cases; unclear when
        a = erfinv(erf(b / sqrt(2)) - 2 * phi_S) * sqrt(2)
        bz = a * tan(pi / 2 - beta)

        gy_r = self.calculate_GRF_correlation_func(l_y, d_y)
        gz_r = self.calculate_GRF_correlation_func(l_z, d_z)
        self.calculate_P_SS_int(gy_r, phi_S, two_point_probability, a)  # calc. two-point correlation function for solid phase
        hz = self.calculate_minimal_hz(phi_A, a, bz)  # self.params['hz_initial'] #
        phi_A, phi_B = self.update_volume_fraction(a, hz, bz, phi_S)# for new hz
        two_point_probability['AA'] = self.calculate_P_XX(self.hn_y_matrix, self.hp_z_matrix, self.integral_matrix, hz, bz, a, gz_r, gy_r, phase='AA')
        two_point_probability['BB'] = self.calculate_P_XX(self.hn_y_matrix, self.hp_z_matrix, self.integral_matrix, hz, bz, a, gz_r, gy_r, phase='BB')

        # depending on the numbers of hermite polynomials used, calc surface area and then approx P_XX might be faster
        # self.calculate_surface_area(a, bz, l_y, d_y, l_z, d_z, beta_angle)
        # self.approximate_low_r_P_XX(6, phi_A, 'AA', two_point_probability)
        # self.approximate_low_r_P_XX(6, phi_B, 'BB', two_point_probability)

        c_r = self.calculuate_electron_density_spatial_correlation_function(two_point_probability, phi_A, phi_B, phi_S)
        intensity, kratky_intensity = self.calculate_intensity(int_const, c_r, porod_const)

        # maybe only for debugging. don't know how multiprocessing writing on the same variable works. but not needed
        # for calculation
        self.results['Intensity'], self.results['two_point_probability'], self.results['phase'] = intensity, \
                                                                                                  two_point_probability, \
                                                                                                  {
                                                                                                      'A': phi_A,
                                                                                                      'B': phi_B,
                                                                                                      'S': phi_S
                                                                                                  }
        self.bz = bz

        return intensity

    def vol_frac_int(self, h, a, bz):
        vol_frac = lambda y, z: 1 / (2 * pi) * exp(-(power(y, 2) + power(z, 2)) / 2)
        return dblquad(vol_frac, self.gaussmin, a, lambda y: h + bz - y * bz / a, self.gaussmin + self.gaussnum, epsabs=1e-10)

    def calculate_volume_beta(self, hh, phi_A, a, bz):
        # vgl. Gomez 2013 volume fraction calculation (SI-28)
        # initialisation
        vol_frac_int = vectorize(self.vol_frac_int)
        volume_fraction_A_int = vol_frac_int(hh, a, bz)[0]

        # ---------- older slower version -----------
        #volume_fraction_A = 0
        #domain = array(
        #    [[1 if zz > (hh + bz - yy * bz / a) and yy < a else 0 for zz in self.coordinate] for yy in self.coordinate])
        #volume_frac_A = sum(sum(domain * self.integral_matrix * self.dz, axis=1) * self.dy)

        #for ii in self.mspace:
        #    yy = self.gaussmin + self.gaussnum * ii / self.m
        #    if yy < a:
        #        z_integral = 0
        #        for jj in self.m_space:
        #            zz = self.gaussmin + self.gaussnum * jj / self.m
        #            if zz >= hh + bz - yy * bz / a:
        #                z_integral += self.integral_matrix[ii][jj] * self.dz
        #
        #        volume_fraction_A += z_integral * self.dy

        return (phi_A - volume_fraction_A_int) ** 2

    def update_volume_fraction(self, a, hz, bz, phi_S):
        vol_frac_int = vectorize(self.vol_frac_int)
        phi_A = vol_frac_int(hz, a, bz)[0]

        # ---------- older slower version -----------

        #volume_frac_A = 0
        #domain = array(
        #    [[1 if zz > (hz + bz - yy * bz / a) and yy < a else 0 for zz in self.coordinate] for yy in self.coordinate])
        #volume_frac_A= sum(sum(domain * self.integral_matrix * self.dz, axis=1) * self.dy)
        # -------- solved it with matrices --------
        #for ii in self.m_space:
        #    yy = self.gaussmin + self.gaussnum * ii / self.m
        #    if yy < a:
        #        z_integral = 0
        #        for jj in self.m_space:
        #            zz = self.gaussmin + self.gaussnum * jj / self.m
        #            if zz >= (hz + bz - yy * bz / a):
        #                z_integral += self.integral_matrix[ii][jj] * self.dz
        #        volume_frac_A += z_integral * self.dy
        return phi_A, 1 - phi_S - phi_A

    def generate_hermite_integral_matrix(self):
        # vgl Gomez (SI-34), integral matrix is the last two terms starting with 1/2*pi

        # hermite arrays
        self.hn_y_matrix = zeros((self.m, self.nmax))
        self.hp_z_matrix = zeros((self.m, self.pmax))
        self.integral_matrix = zeros((self.m, self.m))

        self.fill_hermite_matrix(self.hn_y_matrix, self.nmax)
        self.fill_hermite_matrix(self.hp_z_matrix, self.pmax)

        for ii in self.m_space:
            yy = self.gaussmin + self.gaussnum * ii / self.m
            for jj in self.m_space:
                zz = self.gaussmin + self.gaussnum * jj / self.m
                self.integral_matrix[ii][jj] = 1 / (2 * pi) * exp(-(yy ** 2 + zz ** 2) / 2)

    def fill_hermite_matrix(self, matrix, iter_max):
        # vgl Gomez (SI-34), these are the H_n(x) matrices
        for n in linspace(0, iter_max-1, iter_max, dtype=int):
            for ii in self.m_space:
                yy = self.gaussmin + self.gaussnum * ii / self.m
                herm_poly = hermite(n)
                matrix[ii][n] = power(2, -n / 2) * herm_poly(yy / (2 ** 0.5))

    def calculate_GRF_correlation_func(self, l, d):
        return (1 / cosh(self.r / l)) * (sin(2 * pi * self.r / d) / (2 * pi * self.r / d))

    def calculate_PSD_function(self, l, d):
        psd = zeros(self.num_q)
        for ii in self.q_space:
            psd[ii] = (self.data['q'][ii] * l * d / pi) * (
                        sinh(self.data['q'][ii] * pi * l / 2) *  # maybe backslash \ needed
                        sinh(pi * pi * l / d)) / \
                      (cosh(pi * self.data['q'][ii] * l) + cosh(2 * pi * pi * l / d))

        return psd

    def P_SS_int(self, gy_r, a):
        P_SS = lambda t: 1 / (2 * pi) * 1 / (power(1 - power(t, 2), 0.5)) * exp(-(power(a, 2))/ (1 + t))
        return quad(P_SS, 0, gy_r)

    def calculate_P_SS_int(self, gy_r, phi_S, two_point_probability, a):
        P_SS_int = vectorize(self.P_SS_int)
        two_point_probability['SS'] = P_SS_int(gy_r, a)[0] + power(phi_S, 2)

    # this loop integration is much much slower then the scipy integral, while yielding the same result (err 1e-4)
    def calculate_P_SS(self, gy_r, phi_S, two_point_probability, a):
        # tau is the integration variable for P_SS, see (SI-33) in Gomez 2013
        tau = array([0.00003 * ii for ii in linspace(0, 39999, 40000, dtype=int)])
        d_t = 0.00003
        for jj in self.r_space:
            ii = 0
            if gy_r[jj] > 0:
                while tau[ii] < abs(gy_r[jj]):
                    two_point_probability['SS'][jj] += 1 / (2 * pi) * 1 / (power((1 - power(tau[ii], 2)), 0.5)) * \
                                                            exp(-(power(a, 2)) / (1 + tau[ii])) * d_t
                    ii += 1
            else:
                while tau[ii] < abs(gy_r[jj]):
                    two_point_probability['SS'][jj] -= 1 / (2 * pi) * 1 / (power((1 - power(tau[ii], 2)), 0.5)) * \
                                                            exp(-(power(a, 2)) / (1 + tau[ii])) * d_t
                    ii += 1

            two_point_probability['SS'][jj] += power(phi_S, 2)

    def calculate_P_XX(self, hn_y_matrix, hp_z_matrix, integral_matrix, hz, bz, a, gz_r, gy_r, phase='AA'):
        p_factorials = array([1 / factorial(ii) for ii in linspace(0, self.pmax - 1, self.pmax, dtype=int)])
        n_factorials = array([1 / factorial(ii) for ii in linspace(0, self.nmax - 1, self.nmax, dtype=int)])
        # TODO if pmax /= nmax, need another factorial matrix with nmax

        if phase == 'AA':
            domain = array([[1 if zz > (hz + bz - yy * bz / a) and yy < a else 0 for zz in self.coordinate] for yy in self.coordinate])
        elif phase == 'BB':
            domain = array([[1 if zz <= (hz + bz - yy * bz / a) and yy < a else 0 for zz in self.coordinate] for yy in self.coordinate])
        # Î = D*I
        domain_integral_matrix = domain * integral_matrix
        # Θnp = (Hp x Î) x Hn^T
        Theta_np = matmul(hn_y_matrix.transpose(), matmul(domain_integral_matrix, hp_z_matrix)) * self.dz * self.dy
        Theta_np = power(Theta_np, 2)
        # F_Theta_np = F * Theta_np, where F is the factorial matrix, expanded to 3D matrix for every r
        F = outer(n_factorials, p_factorials)
        F_Theta_np_r = array([F * Theta_np for rr in self.r_space])

        # G is a 3D matrix containing the outer product of gy_r and gz_r raised to the power of n, p for every radius
        G = array([outer(array([power(gy_r[rr], n) for n in self.n_space]), array([power(gz_r[rr], p) for p in self.p_space]))
                   for rr in self.r_space])

        # now for every r, we calculate the product of the gy_r, gz_r raised to the power of n, p to combine
        # it with the corresponding (n,p) element of the Fe_Theta_np matrix, then we sum up all the 2D matrix along n and p for
        # every r and receive P_XX of dimension r

        G_F_Theta_np_r = G * F_Theta_np_r
        P_XX = G_F_Theta_np_r.sum(axis=(1, 2))
        return P_XX

    def calculate_minimal_hz(self, phi_A, a, bz):
        x0 = array([self.params['hz_initial']])
        res = minimize_parallel(fun=self.calculate_volume_beta, x0=x0, args=(phi_A, a, bz), bounds=[(-2.5, 2)], tol=1e-6)

        return res.x[0]

    def calculate_surface_area(self, a, bz, l_y, d_y, l_z, d_z, beta_angle):
        # vgl. Gomez 2013 (SI-27) and (SI-54) to (SI-57)
        fy = self.calculate_PSD_function(l_y, d_y)
        fz = self.calculate_PSD_function(l_z, d_z)
        l_y_squared_inverse = 0
        l_z_squared_inverse = 0

        for ii in linspace(0, self.num_q - 2, self.num_q - 1, dtype=int):
            l_y_squared_inverse += 1 / 6 * self.data['q'][ii] * fy[ii] * (self.data['q'][ii + 1] - self.data['q'][ii])
            l_z_squared_inverse += 1 / 6 * self.data['q'][ii] * fz[ii] * (self.data['q'][ii + 1] - self.data['q'][ii])

        l_y_updated = 1 / sqrt(l_y_squared_inverse)
        l_z_updated = 1 / sqrt(l_z_squared_inverse)
        b_normal = (bz + self.hz) * cos(pi / 2 - beta_angle)

        s_ab = sqrt(2) / pi * sqrt((cos(beta_angle) / l_y_updated) ** 2 +
                                            (sin(beta_angle) / l_z_updated) ** 2) * exp(
            -(b_normal ** 2) / 2) * \
               (1 - erf((b_normal * cos(beta_angle) - a) / (sqrt(2) * sin(beta_angle))))
        s_as = sqrt(2) / pi * exp(-(a ** 2) / 2) / l_y_updated * \
               (1 - erf((b_normal - a * cos(beta_angle)) / (sqrt(2) * sin(beta_angle))))
        s_bs = sqrt(2) / pi * exp(-(a ** 2) / 2) / l_y_updated * \
               (1 - erf((b_normal + a * cos(beta_angle)) / (sqrt(2) * sin(beta_angle))))

        self.surface_area['S'] = s_as + s_bs
        self.surface_area['A'] = s_as + s_ab
        self.surface_area['B'] = s_bs + s_ab

    def approximate_low_r_P_XX(self, max_r, phase, phase_index, two_point_probability):

        for ii in linspace(0, max_r - 1, max_r, dtype=int):
            # set 0 to erase previously calculated values
            two_point_probability[phase_index][ii] = 0
            two_point_probability[phase_index][ii] = phase - (self.surface_area[phase_index[0]] / 4) * self.r[ii]

    def calculuate_electron_density_spatial_correlation_function(self, two_point_probability, phi_A, phi_B, phi_S):
        # vgl Gomez 2013 (SI-4) electron density spatial correlation function
        c_r = (self.params['rho_S'] - self.params['rho_A']) * (self.params['rho_S'] - self.params['rho_B']) * (
                    two_point_probability['SS'] - phi_S ** 2) + \
                   (self.params['rho_A'] - self.params['rho_S']) * (self.params['rho_A'] - self.params['rho_B']) * (
                               two_point_probability['AA'] - phi_A ** 2) + \
                   (self.params['rho_B'] - self.params['rho_S']) * (self.params['rho_B'] - self.params['rho_A']) * (
                               two_point_probability['BB'] - phi_B ** 2)
        c_r -= c_r[-1]

        return c_r

    def calculate_intensity(self, int_cons, c_r, porod_const):
        # vlg. Gomez 2013 (SI-2)
        intensity = zeros(self.num_q)
        for ii in self.q_space:
            for jj in linspace(0, self.num_r - 2, self.num_r - 1, dtype=int):
                intensity[ii] += sin(self.data['q'][ii] * self.r[jj]) / (self.data['q'][ii] * self.r[jj]) * c_r[jj] * 4 * pi * \
                                      (1e-7 * self.r[jj]) ** 2 * (self.r[jj + 1] - self.r[jj]) * 1e-7

        # unclear, if Porod subtration is necessary for SANS
        intensity += porod_const / power(self.data['q'], 4)
        intensity *= int_cons
        kratky_intensity = power(self.data['q'], 2) * intensity

        return intensity, kratky_intensity

    # just in case one would like to load new data
    def load_new_data(self, df):
        self.data = {
            'q': df.Mod_Q.to_numpy(),
            'Intensity': df.Intensity.to_numpy(),
            'Error': df.Err_I.to_numpy()
        }
        self.num_q = len(self.data['q'])

    def reset_calculations(self):
        self.two_point_probability = {
            'SS': zeros(self.num_q),
            'AA': zeros(self.num_q),
            'BB': zeros(self.num_q)
        }
        self.c_r = None
        self.intensity = zeros(self.num_q)

