# Assay parameterization simulations

This sub-repository contains all code necessary to perform the assay parameterization simulations described in *Le, et al.*. This repository contains files to aid in this analysis:

- `MCsimulation.R` - Performs Monte Carlo-based simulation of the sequencing process to provide quantitative accuracy estimates for a given spread of ddG values, assuming a normal distribution. This file is capable of simulating the entire 4^10 species library, as it does not perform equilibrium binding simulations and assumes that ligand depletion is not a problem.
- `simbind.py` - Perform full equilibrium binding simulation for sampled sequences from a larger distribution, following by Monte Carlo-based simulation of the sequencing process. This simulator takes into account ligand depletion/competition by performing full binding simulations, but is incapable of simulating the full library (simulations limited to ~1000 simulated species max).
- `EnergeticRangeAnalysis.Rmd` - A full analysis of the energetic ranges for which equilibrium binding measurements based on sequenced input library would be valid.