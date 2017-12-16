__author__ = "Scott Longwell and Tyler Shimko"
__credits__ = ["Scott Longwell", "Tyler Shimko"]
__license__ = "MIT"
__maintainer__ = "Tyler Shimko"
__email__ = "tshimko@stanford.edu"

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import *
from itertools import product
from joblib import Parallel, delayed 


class Library():
    """
    Defines the molecular library used in the experiment

    Args:
    - concs (array-like) - concentrations of all substrates species
    - k_on (array-like) - values of k_on
    - k_off (array-like) - values of k_off
    - ddG_range (float) - spread of the library in ddG (kcal/mol)
    - dummy (int) - position of the dummy (high-concentration) substrate
    """

    def __init__(self, concs, k_on, k_off, ddG_range, total_seqs, dummy): 
        self.conc = sum(concs)
        self.seqs = pd.DataFrame()
        self.seqs['input_conc'] = concs # moles
        self.seqs['k_on'] = k_on # 1/(M*sec)
        self.seqs['k_off'] = k_off # 1/sec

        self.seqs['k_a'] = self.seqs.k_on/self.seqs.k_off # 1/M
        self.seqs['k_d'] = self.seqs.k_off/self.seqs.k_on # M
        self.seqs['ddG_range'] = ddG_range
        self.seqs['total_species'] = total_seqs
        self.seqs['total_input_conc'] = np.sum(concs)
        self.seqs['dummy'] = dummy
        self.seqs['dummy_bool'] = False
        self.seqs.at[dummy, 'dummy_bool'] = True
        
    def __repr__(self):
        pd.set_option('display.float_format', '{:.2E}'.format)
        return str(self.seqs.describe(percentiles=[]).drop('count', axis=0))
        
    def plot(self):
        """
        Plot distributions of library properties
        """

        sns.set_context('notebook')
        
        # Plot histogram of concentrations
        fig, axes = plt.subplots()
        sns.distplot(self.seqs.input_conc)
        axes.get_yaxis().set_visible(False)
        plt.show()
        
        # Plot distribution of k_on, k_off
        fig, axes = plt.subplots(ncols=2)
        plt.tight_layout()
        fig.tight_layout(w_pad=3)
        sns.distplot(self.seqs.k_on, ax=axes[0])
        sns.distplot(self.seqs.k_off, ax=axes[1])
        for ax in axes:
            ax.ticklabel_format(axis='y',style='sci',scilimits=(1,4))
        plt.show()
        
        # plot distribution of k_a, k_d
        fig, axes = plt.subplots(ncols=2)
        plt.tight_layout()
        fig.tight_layout(w_pad=3)
        sns.distplot(self.seqs.k_a, ax=axes[0])
        sns.distplot(self.seqs.k_d, ax=axes[1])
        for ax in axes:
            ax.ticklabel_format(axis='y',style='sci',scilimits=(1,4))
        plt.show()

def create_lib(n_seqs, ddG_range, total_seqs=4**10, input_conc=1e-6, dummy=50):
    """
    Define the library parameters (K_d, k_on, k_off) for use in an equilibrium
    binding simulation. This function assumes an initial library concentration
    and total number of species in the library. However, the function only
    generates a small, user-defined number of species upstream and downstream
    of a dummy point used to simulate the high-density portion of the library.
    This strategy makes simulations tractable as it is not computationally
    feasible to simulate libraries > ~10000 species.
    
    Args:
    - n_seqs (int) - total number of sequences in the simulated library
    - ddG_range (float) - spread of the library in ddG (kcal/mol)
    - total_seqs (int) - total number of sequences in the "true" library, used
        to establish the concentration of the dummy species (default = 4^10)
    - input_conc (float) - concentration of the total library (default = 1e-6)
    - dummy (int) - position of the "dummy" high-concentration species in the
        simulated library, changes shape of distribution (default = 50)

    Returns:
    - Library with evenly spaced ddG substrates spread across the prescribed
        energetic range
    """

    # Define the total number of sequences
    # One sequence will be a "dummy" representing the median affinity
    # portion of the distribution
    n_seqs += 1

    # Get the number of low concentration species and their respective
    # concentrations in the full library
    low_conc_seqs = n_seqs - 1
    ind_sp_conc = input_conc / total_seqs

    # Make the concentration vector
    concs = np.empty(n_seqs)
    concs.fill(ind_sp_conc)

    # Get the concentration of "dummy" species
    mid_conc = input_conc - (low_conc_seqs * ind_sp_conc)

    # Generate the K_Ds
    ddG = ddG_range / 2
    min_k_d = np.log10(1 / np.exp((-9 - ddG) / -.593))
    max_k_d = np.log10(1 / np.exp((-9 + ddG) / -.593))
    
    k_ds = np.logspace(min_k_d, max_k_d, num=101)

    # Set the concentration of the median one to the larger conentration
    concs[dummy] = mid_conc

    # Set all k_ons to 1e7
    k_on = np.empty(n_seqs)
    k_on.fill(1e7) # 1/sec

    # Calculate k_offs
    k_off = k_ds * k_on
    
    return Library(concs, k_on, k_off, ddG_range, total_seqs, dummy)


class Assay():
    """
    Define and run an assay using a given library

    Args:
    - n_seqs (int) - total number of sequences in the simulated library
    - ddG_range (float) - spread of the library in ddG (kcal/mol)
    - total_seqs (int) - total number of sequences in the "true" library, used
        to establish the concentration of the dummy species (default = 4^10)
    - input_conc (float) - concentration of the total library (default = 1e-6)
    - dummy (int) - position of the "dummy" high-concentration species in the
        simulated library, changes shape of distribution (default = 50)
    - protein_conc (float) - concentration of protein in assay
        (default = 30e-9)
    """

    def __init__(self, n_seqs, ddG_range, total_seqs=4**10, input_conc=1e-6,
                 dummy=50, protein_conc=30e-9):

        self.library = create_lib(n_seqs, ddG_range, total_seqs, input_conc,
                                  dummy)
        self.protein_conc = protein_conc
        
    def __repr__(self):
        s  = 'sites/lib: {}\tlib/sites: {}'.format(self.protein_conc/self.library.conc,
                                                   self.library.conc/self.protein_conc)
        s += "\n"
        s += 'sites/seq: {}\tseq/sites: {}'.format(self.protein_conc/self.library.seqs.concs.mean(),
                                                   self.library.seqs.concs.mean(), self.protein_conc)
        return s
    
    def simulate(self, time):
        """
        Simulate a number of seconds as the reaction moves to equilibrium

        Args:
        - time (int) - duration (in seconds) to simulate 
        """
        seqs = self.library.seqs.copy()

        N = len(seqs)
        S_slice  = slice(0,-1,2)   # i={0,2,4,...,2N-2} 2N+1 species   even elements, less last one
        PS_slice = slice(1,None,2) # i={1,3,5,...,2N-1} 2N+1 species   odd elements 
        P_slice  = slice(-1,-2,-1) # i={2N}             2N+1 species   last element

        ron_slice = slice(0,None,2)  # i={0,2,4,...,2N-2} 2N reactions   even elements 
        roff_slice = slice(1,None,2) # i={1,3,5,...,2N-1} 2N reactions   odd elements

        # construct k: rate constants
        # 2N rates
        # kon1, koff1, kon2, koff2,...
        k = np.zeros(2 * N)
        k[ron_slice] = seqs['k_on']
        k[roff_slice] = seqs['k_off']

        # construct y_0 (initial concentrations)
        y_0 = np.zeros(2*N+1)
        y_0[S_slice] = seqs.input_conc
        y_0[PS_slice] = 0.0
        y_0[P_slice] = self.protein_conc

        # dy_t: vector, 1 element -> 1 species
        def dy_t(y_t, t):
            # y_t: state (concentration) vector at t
            # r_t: reaction rate vector at t
            r_t = k*y_t[:-1] # i.e. y_t[S_slice] and y_t[PS_slice] 
            r_t[ron_slice] *= y_t[P_slice]
            
            dy_t = np.empty_like(y_t)
            dy_t[S_slice] = - r_t[ron_slice] + r_t[roff_slice]
            dy_t[PS_slice] =  r_t[ron_slice] - r_t[roff_slice]
            dy_t[P_slice] = - r_t[ron_slice].sum() + r_t[roff_slice].sum()
            return dy_t
        
        self.t_ls = np.linspace(0, time, 500)

        # y(t,species)
        y, info = sp.integrate.odeint(dy_t, y_0, self.t_ls, hmax=1, full_output=True)

        self.info = info

        self.PS = y[:,PS_slice]
        self.S = y[:,S_slice]
        self.P = y[:,P_slice]

        seqs['bound_conc'] = self.PS[-1]
        seqs['unbound_conc'] = self.S[-1]

        self.bind_sim_results = seqs

    def sample_sequences(self, depths, reps=1):
        """
        Simulate the stochastic sampling process of sequencing

        Args:
        - depths (list) - list of sequencing depths to use for each sublibrary
        - reps (int) - number of replicate experiments to run
        """

        self.sequencing_results = self.bind_sim_results.copy()
        self.sequencing_results['seq_number'] = self.sequencing_results.index.values
        seq_results = self.bind_sim_results.copy()

        # Get the probabilities of being drawn
        bound_p = seq_results['bound_conc']/np.sum(seq_results['bound_conc'])
        unbound_p = seq_results['unbound_conc']/np.sum(seq_results['unbound_conc'])
        input_p = seq_results['input_conc']/np.sum(seq_results['input_conc'])
        
        depths = [int(depth) for depth in depths]

        read_samples = []

        # Loop thrpugh the desired depths
        for depth in depths:

            # Perform every replicate
            for i in range(reps):
                
                # Sample each library
                bound_counter = Counter(np.random.choice(seq_results.index.values,
                                                         size=depth,
                                                         p=bound_p))

                bound_counts = [bound_counter[i] if i in bound_counter.keys()
                                else 0 for i in range(len(seq_results))]

                unbound_counter = Counter(np.random.choice(seq_results.index.values,
                                                           size=depth,
                                                           p=unbound_p))

                unbound_counts = [unbound_counter[i] if i in
                                  unbound_counter.keys() else 0 for i in
                                  range(len(seq_results))]

                input_counter = Counter(np.random.choice(seq_results.index.values,
                                                         size=depth,
                                                         p=input_p))

                input_counts = [input_counter[i] if i in input_counter.keys()
                                else 0 for i in range(len(seq_results))]
                
                # Add the columns to the data frame              
                read_sample = pd.DataFrame()

                read_sample['seq_number'] = seq_results.index.values
                read_sample['depth'] = depth
                read_sample['rep'] = i
                read_sample['bound_count'] = bound_counts
                read_sample['unbound_count'] = unbound_counts
                read_sample['input_count'] = input_counts

                read_samples.append(read_sample)

        read_samples = pd.concat(read_samples)
                
        self.sequencing_results = self.sequencing_results.merge(read_samples,
                                                                how='outer',
                                                                on='seq_number')

    def plot_curves(self):
        """
        Plot the concentrations of PS, S, P over time
        """

        # PS(t,ps_i)
        plt.plot(self.t_ls/60, self.PS, lw = .2, color='r')
        plt.title('[PS]')
        plt.xlabel('time [min]')
        plt.yscale('log')
        plt.show()

        # S(t,s_i)
        plt.plot(self.t_ls/60, self.S, lw = .2, color='r')
        plt.title('[S]')
        plt.xlabel('time [min]')
        plt.yscale('log')
        # plt.ylim((1e-16, 1e-12))
        plt.show()

        # P(t)
        plt.plot(self.t_ls/60, self.P, lw = .2, color='r')
        plt.title('[P]')
        plt.xlabel('time [min]')
        plt.yscale('log')
        plt.show()

    def plot_results(self):
        """
        Plot the results of the estimated K_ds vs. the actual K_ds 
        """
        t = -1 

        kd_actual = self.library.seqs['k_d']

        d = {'(P*S)/PS = K_d': self.P[t]*self.S[t]/self.PS[t],
             'S/PS'      : self.S[t]/self.PS[t],
             
             'in/PS'     : self.S[0]/self.PS[t],
             'in_avg/PS' : self.S[0].mean()/self.PS[t],
             
             'S/in'      : self.S[t]/self.S[0],
             'S/in_avg'  : self.S[t]/self.S[0].mean()}
            
        # Print stats and create plot
        print('P=[P], S=[S], PS=[PS] \n')
        for title, kd_observed in d.items():
            slope, intercept, r_value, p_value, std_err = sp.stats.linregress(kd_observed, self.library.seqs['k_d'])
            print('slope:', slope, ' int:', intercept)
            print('r^2:', r_value**2)
            plt.title(title)
            plt.scatter(kd_actual, kd_observed, s=2.5)
            plt.xlabel('kd_actual')
            plt.ylabel('kd_observed')
            plt.xlim(kd_actual.min(),kd_actual.max())
            plt.ylim(kd_observed.min(),kd_observed.max())
            plt.xscale('log')
            plt.yscale('log')
            plt.show()

if __name__ == '__main__':
    # Set the number of cores to use for parallel simulations
    N_CORES = 35

    # Set the parameters for the assay simulations
    ddG_ranges = [0.5, 1, 2, 4, 8, 16]
    total_seqs = [4**x for x in [4, 6, 8, 10]]
    dummy_positions = [50, 75]

    # Get all combinations of assay parameters
    combinations = list(product(ddG_ranges, total_seqs, dummy_positions))

    # Define a function to parse a combination and run the individual sims
    def run_assay(arg):
        ddG_range, total_seq, dummy = arg
        
        assay = Assay(100, ddG_range, total_seqs=total_seq, dummy=dummy)
        assay.simulate(3600)
        
        seq_depths = [1e6, 1e7, 2.5e7, 5e7]
        
        assay.sample_sequences(seq_depths, reps=5)
        
        return assay.sequencing_results

    # Run assays in parallel and save data
    outputs = Parallel(n_jobs=N_CORES)(map(delayed(run_assay), combinations))

    output = pd.concat(outputs)

    output.to_csv('../data/complete_simulations.csv', index=False)

