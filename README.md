# BET-seq analysis repository

This repository contains all the code to perform the analyses described in the *Le, et al.* manuscript describing MITOMI sequencing-based analysis of Pho4 and Cbf1 core binding site flanking sequences (BET-seq). This pre-print of the manuscript can be found on bioRxiv [here](https://www.biorxiv.org/content/early/2017/09/26/193904.article-metrics) and the final PNAS published version can be found [here](https://doi.org/10.1073/pnas.1715888115).

The following files are in this root directory:

- `GEO_submission.xls` - Excel spreadsheet with information for GEO submission of raw datasets
- `setup.R` - Code to setup necessary datasets and variables for full analysis
- `Main_Figures.Rmd` - Code to generate all of the figures in the main text
- `SI.Rmd` - Code to generate all figures and tables in the supporting information
- `README.md` - This README
- `data` - All of the processed data files (not included in this repo, data files can be found on [Figshare](https://figshare.com/articles/BET-seq_Processed_Data/5728467))
- `data_processing.py` - Script to process raw fastq files
- `figures` - All of the final typeset figures from the main text
- `images` - All of the subplot/subfigure images
- `neural_network` - Code to run the neural network model
- `simulations` - Code to run assay parameterization simulations
