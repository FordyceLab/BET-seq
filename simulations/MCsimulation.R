#################################################
### Required packages ###########################

library(foreach)
library(dplyr)
library(doMC)

#################################################
### Sequencing-based binding assay parameters ###

# Affinity range (ddG, kcal/mol)
iVal = 1:2

# Substrate library size (# of sequences)
jVal = 10^(2:3)

# Sequencing depth (# of reads)
kVal = 10^(3:4)

# Simulation replicates
lVal = 1:10

# Output directory
outFile = "~/Desktop/test.txt"

# Number of cores for parallel computation
nCores = 2

#################################################
### Initialize dependent variables ##############

# Calculate number of simulated conditions for Bonferroni p-value correction
testCount = lapply(list(mVal,nVal,iVal,jVal),function (x) x %>% length()) %>%
  unlist() %>%
  prod()

# Allocate cores
registerDoMC(cores = nCores)

#################################################
### Perform simulation ##########################

returnDf =
  foreach(i = iVal, .combine = "rbind")%do%{ # iterate affinity ranges
    gibbsDiff = i
    foreach(j = jVal, .combine = "rbind")%do%{ # iterate library sizes
      dnaSpecies = j
      foreach(k = kVal, .combine = "rbind")%do%{ # iterate sequencing depths
        libDepth = k
        # Library distribution assumptions:
        # 1) normality
        # 2) zero-centered
        # 3) thermodynamic scale = RT at 298 K
        dnaRatios = sapply(rnorm(dnaSpecies, mean = 0, sd = gibbsDiff/6), function(x) exp(-x/.593))
        print(paste(i,j,k))
        foreach(l = lVal, .combine = "rbind")%dopar%{ # iterate simulation replicates
          trueRatioDf =
            tbl_df(data.frame(specIndex = 1:dnaSpecies, trueRatio = dnaRatios)) %>%
            mutate(prob = trueRatio/sum(trueRatio))
          obsRatioDf =
            # Sequencing depth assumption: equal read allocation between bound and input fractions
            inner_join(tbl_df(data.frame(sampleBound = sample(trueRatioDf$specIndex, libDepth/2, prob = trueRatioDf$prob, replace = T))) %>%
              count(sampleBound) %>%
              rename(specIndex = sampleBound, bound = n),
              tbl_df(data.frame(sampleUnbound = sample(trueRatioDf$specIndex, libDepth/2, prob = rep(1/dnaSpecies,dnaSpecies), replace = T))) %>%
              count(sampleUnbound) %>%
              rename(specIndex = sampleUnbound, unbound = n)) %>%
            mutate(obsRatio = bound/unbound)
          ratioDf =
            inner_join(trueRatioDf %>% select(-prob), obsRatioDf %>% select(-bound, -unbound))
          # Correlation assumption: If Bonferroni-corrected p-value > 0.05, then correlation coefficient = 0
          rSqd =
            tryCatch({ifelse(cor.test(ratioDf$trueRatio, ratioDf$obsRatio, method = "pearson")$p.value < (0.05/testCount), cor.test(ratioDf$trueRatio, ratioDf$obsRatio, method = "pearson")$estimate, NA)}, error = function(err) {NA})
          # Output description:
          # readTotal = sequencing read depth
          # meanRead_perSeq = mean sequencing reads per substrate species
          # iter = simulation replicate
          # r2 = correlation coefficient (rho or r^2)
          # seqObsPct = fraction of observed substrates
          # acc = accuracy (defined as r2 * fraction of observed substrates)
          data.frame(readTotal = libDepth, 
                     meanRead_perSeq = (libDepth/2)/dnaSpecies, 
                     iter = j, 
                     r2 = ifelse(is.na(rSqd),0,rSqd^2), 
                     ddG = gibbsDiff, libSize = dnaSpecies, 
                     seqObsPct = nrow(obsRatioDf)/dnaSpecies) %>%
                      mutate(acc = r2 * seqObsPct)
        }
      }
    }
  }

write.table(returnDf, outFile, col.names = T, row.names = F, quote = F)