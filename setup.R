# Packages
library(xtable)
library(latex2exp)
library(ggseqlogo)
library(RColorBrewer)
library(glmnet)
library(knitr)
library(utils)
library(seqinr)
library(cowplot)
library(data.table)
library(ggplot2)
library(hexbin)
library(foreach)
library(tidyr)
library(broom)
library(MASS)
library(stringr)
library(dplyr)

# Global aesthetics
presentation <-  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5,
                                  family = "Arial",
                                  face = "bold",
                                  size = 12),
        axis.title = element_text(family = "Arial",
                                  face = "bold",
                                  size = 6),
        axis.text = element_text(family = "Arial",
                                 face = "bold",
                                 size = 6),
        legend.text = element_text(family = "Arial",
                                   face = "bold",
                                   size = 6),
        legend.title = element_text(family = "Arial",
                                    face = "bold",
                                    size = 6),
        strip.text = element_text(family = "Arial",
                                  face = "bold",
                                  size = 6),
        strip.text.x = element_text(vjust = 0))

# Functions

revcomp = function(string) {
  # Input: string = character string of DNA
  # Output: character string of reverse complement DNA
  fwdString = chartr("ACTG_", "TGAC_", string) %>% strsplit(.,NULL)
  revString = rev(fwdString[[1]])
  returnString = paste(revString, collapse = "")
  return(returnString)
}

addMotif = function(df, var) {
  # Input: df = dataframe of ddG values with accompanying experimental parameters; 
  # var = character string identifying column name of ddG values;
  # Output: dataframe of mean ddG values grouped by nucleotide identity and position
  monuc.flank = data.frame(side = factor(c(rep("Upstream",5),rep("Downstream",5)), 
                                         levels = c("Upstream","Downstream")), 
                           monuc.pos = factor(paste0(c(-5:-1, 1:5),sep=""), levels = c(-5:-1, 1:5)), 
                           monuc.key = paste0("X",sprintf("%02d", 1:10),sep=""))
  tmp = 
    df %>%
    rename_(target=var) %>%
    gather(monuc.key, monuc.value, X01:X10) %>% 
    group_by(monuc.key,monuc.value,rep,protein) %>%
    summarise(monuc.meanval = mean(target, na.rm = T)) %>%
    ungroup() %>% 
    left_join(.,monuc.flank,by="monuc.key")
}

createCurves = function(df) {
  # Input: df = dataframe for raw binding curve measurements
  # Output: dataframe of fitted curve values
  
  # return global max
  maxVal = nls(fBound ~ (maxBound * Conc) / (Conc + Kd[index]),
               data=df,
               start=list(maxBound = max(df$fBound), 
                          Kd = rep(0.5 * max(df$Conc), max(df$ID))),
               control = nls.control(printEval = FALSE, warnOnly = T)) %>%
    tidy() %>%
    filter(term == "maxBound") %>%
    .$estimate
  
  # all individual fits
  pho4Fit = 
    foreach(i = unique(df$ID), .combine = "rbind")%do%{
      print(i)
      tryCatch({
        curveSlice = df %>% filter(ID == i)
        mod = nls(fBound ~ (maxVal * Conc) / (Conc + Kd),
                  data=curveSlice,
                  start=list(Kd = 0.5 * max(df$Conc)))
        
        modVal = tidy(mod)
        curveSlice %>% 
          select(-index, -ID) %>%
          left_join(data.frame(.fitted = predict(mod, data.frame(Conc = unique(df$Conc))), Conc =unique(df$Conc)),.) %>%
          mutate(term = "Kd") %>%
          left_join(.,modVal) %>%
          mutate(index = paste("ID", i, sep = ""))
      }, error = function(err) {})
    } %>% 
    tbl_df()
  return(pho4Fit)
}

createTerms = function(termLim) {
  # Input: termLim = integer value for number of flanking positions interacting
  # Output: dataframe of linear regression terms
  foreach(i = 1:termLim, .combine = "rbind")%do%{
    Vars =
      expand.grid(rep(list(1:10),i)) %>%
      tbl_df() %>%
      filter_(.dots = if(i > 1) {paste0("Var",2:i,">Var",1:(i-1))}) # remove redundancy
    
    if(i > 1) {nnVars = Vars %>% 
      filter_(.dots = paste0("Var",2:i,"-Var",1:(i-1),"==1")) %>% 
      filter_(.dots = paste0("Var1 != ", (7-i):5)) }else{
      nnVars = NA} # create nearest neighbor groups
    
    termsList = list(Vars, nnVars) # combine variable dataframes
    termVars = foreach(j = if(!is.data.frame(nnVars)){1}else{1:2}, .combine = "rbind")%do%{ # iterate through each dataframe
      varRows = termsList[[j]] 
      foreach(k = 1:nrow(varRows), .combine = "rbind")%do%{ # create variables from rows
        returnVar =
          varRows %>%
          slice(k) %>%
          gather(key,value,everything()) %>%
          mutate(value = paste("X",sprintf("%02d", value), sep="")) %>%
          .$value %>%
          paste0(.,collapse=":")
        data.frame(Var = returnVar, terms = i, nearN = j-1) # assign parameters
      }
    }
  }
}

createPred = function(termDf, inputDf, testDf) {
  # Input: termDf = dataframe of flanking positions interacting; 
  # fileName = character string specifying output file name
  # inputDf = dataframe of binding data (filter on single protein)
  # testDf = dataframe of titration data
  # Output: list of dataframes containing fitted values vs empirical
  
  termGroups = distinct(termDf,terms,nearN) # define model classes
  trainSet = inputDf
  testSet = testDf
  
  tmp = foreach(i = 1:nrow(termGroups), .combine = "rbind")%do%{ # iterate through classes
    
    termClass = termGroups %>% slice(i)
    modelName = paste0(ifelse(termClass$nearN == 1,"Nearest-",""),termClass$terms,"-nuc")
    mod = lm(formula(paste("ddG ~ ", paste0(left_join(termClass,termDf) %>% .$Var, collapse = "+"), sep = "")), data = trainSet) # linear regression
    
    data.frame(.fitted = predict.lm(mod, newdata = testSet)) %>% 
      mutate(model = modelName) %>%
      bind_cols(.,testSet %>% select(flank,estimate,std.error,stdE))
  }
  return(tmp)
}

contr.Dummy <- function(contrasts, ...){
  # Summary: sets options for categorical regression to eliminate the intercept term for simpler interpretation
  # Inputs: constrasts = contrasts objects originating within a model object
  # Outputs: contrasts options set to eliminate intercept term
  
  conT <- contr.treatment(contrasts=FALSE, ...)
  conT
  }

addVar = function(listDf, charFlank, intVal) {
  # Inputs: listDf = list of position df's from global env.
  # charFlank = chr string of flank sequence
  # intVal = int specifying max number of positions
  # Output: chr string of concatanated variables
  sapply(1:intVal, function (y) {
    dfPos = listDf[[y]] 
    
    string = strsplit(charFlank,"")[[1]]
    
    dfPos  %>% 
      mutate_each(funs(sapply(., function(x) string[x])), everything()) %>% # extract bases at each position
      unite(var,everything(),sep="") %>%
      .$var %>%
      paste0(.,collapse=".")
    }) %>% paste0(.,collapse=".")
  }

convertFlank = function(charTerm, intRow, listDf) {
  # Inputs: charTerm = chr string of bases in a term
  # intRow = int specifying row in listDf
  # listDf = list of df's from global env that contain interacting positions in flank
  # Output: chr string of full flank
  returnString = "NNNNNNNNNN"
  intVal = nchar(charTerm)
  dict = data.frame(row = 1:sum(sapply(1:intVal, function(x) choose(10,x))), 
                    listIndex = rep(1:intVal, sapply(1:intVal, function(x) choose(10,x))),
                    withinRow = sapply(1:intVal, function(x) 1:choose(10,x)) %>% unlist())
  dictIndex = dict %>% filter(row == intRow)
  lettPos = listDf[[dictIndex$listIndex]] %>% 
    slice(dictIndex$withinRow) %>% 
    unlist() %>% 
    as.vector()
  foreach(i = 1:dictIndex$listIndex)%do%{
    substring(returnString,lettPos[i],lettPos[i]) <- substring(charTerm,i,i)
  }
  return(returnString)
}

makePairs <- function(data, term) {
  # Inputs: data = dataframe of all col to cross pair + terms col
  # Output: dataframe that contains cross pairs
  tmp = data %>% select_(paste0("-",term))
  termVec = data[,term]
  grid <- expand.grid(x = 1:ncol(tmp), y = 1:ncol(tmp))
  grid <- subset(grid, x != y)
  all <- do.call("rbind", lapply(1:nrow(grid), function(i) {
    xcol <- grid[i, "x"]
    ycol <- grid[i, "y"]
    data.frame(xvar = names(tmp)[ycol], yvar = names(tmp)[xcol], 
               x = tmp[, xcol], y = tmp[, ycol], tmp)
  }))
  all$xvar <- factor(all$xvar, levels = names(tmp))
  all$yvar <- factor(all$yvar, levels = names(tmp))
  return(tbl_df(all) %>% mutate(term = rep(termVec, length=nrow(.))))
}

encodeTerms = function(df, bsize, classVal, posDf) {
  # Inputs: df = dataframe of flank and ddG
  # bsize = int specifying batch size
  # classVal = int specifying interacting positions
  # posDf = list of dataframes of positions
  # Outputs: dataframe of model variables
  batch = bsize
  tmpDf = df
  iter = ceiling(nrow(df)/bsize)

  foreach(i = 1:iter, .combine = "rbind")%dopar%{
    start = ((i-1)*batch)+1
    stop = i*batch
    tmpDf %>% 
      slice(start:stop) %>%
      select(flank, ddG) %>%
      group_by(flank) %>%
      mutate(varSet = addVar(posDf,flank,classVal)) %>%
      separate(varSet,paste0("Var",1:sum(sapply(1:classVal, function(x) choose(10,x))),"."),sep="[.]") %>%
      ungroup() %>% 
      mutate_each(funs(as.factor),contains("Var"))
  }
}

### DMC Functions
extractLine = function(oneLine) { 
  # Input: Character string of read line
  # Output: Dataframe of DMC variables
  # Line arrangement: NW.Var1 = V1, NW.Var2 = V2, SE.Var1 = V3, SE.Var2 = V4
  
  strsplit(oneLine, "\t") %>% 
    unlist() %>%
    matrix(.,1,4) %>%
    as.data.frame()
}

extractBase = function(seqStr) {
  # Input: Character string of single DMC variable
  # Output: Dataframe of base and position 
  
  reStr = gregexpr("[ACGT]", seqStr)
  matchStr = regmatches(seqStr,reStr) %>% unlist()
  varStr = paste0("Var", reStr %>% unlist())
  return(data.frame(pos = varStr, base = matchStr) %>% spread(pos,base))
}

extractSite = function(df) {
  # Input: Dataframe of DMC assingments
  # Ouput: Dataframe of DMC sites, bases and positions
  
  bind_rows(
    bind_cols(extractBase(as.character(df[1,1])),extractBase(as.character(df[1,2]))) %>% mutate(site = "NW"),
    bind_cols(extractBase(as.character(df[1,1])),extractBase(as.character(df[1,4]))) %>% mutate(site = "NE"),
    bind_cols(extractBase(as.character(df[1,3])),extractBase(as.character(df[1,4]))) %>% mutate(site = "SE"),
    bind_cols(extractBase(as.character(df[1,3])),extractBase(as.character(df[1,2]))) %>% mutate(site = "SW"))
}

combineBg = function(df) {
  # Input: Dataframe of DMC sites, bases and positions
  # Output: Dataframe of DMC sites, bases, positions, backgrounds
  
  left_join(df, expand.grid(rep(list(c('A', 'G', 'T', 'C')), 10))) %>% tbl_df()
}

readBind = function(sourceDf) {
  # Arg: sourceDf = df of seq measurements;
  # Output: Dataframe of binding energies, positions, bases
  
  df = sourceDf %>%
    select(X01:X10, ddG) %>%
    tbl_df()
  colnames(df) = c(paste0("Var", 1:10), "ddG")
  return(df)
}

fillDmc = function(df, dmcVar) {
  # Input: Dataframe of energies, positions, and bases; Dataframe of dmc variables
  # Output: Dataframe of DMC energies
  dropCol = colnames(dmcVar %>% select(-site))
  df %>%
    select_(.dots = sapply(paste0("-", dropCol), . %>% {as.formula(paste("~", .))})) %>%
    spread(site, ddG) %>%
    na.omit()
}

ttestDmc = function(df) {
  # Input: Dataframe of DMC energies
  # Output: Dataframe of t-test results
  
  bind_rows(glance(t.test(df$N, df$S)) %>% mutate(type = "Edge"), 
            glance(t.test(df$SE, df$NW)) %>% mutate(type = "SE-NW"), 
            glance(t.test(df$NE, df$NW)) %>% mutate(type = "NE-NW"), 
            glance(t.test(df$SW, df$NW)) %>% mutate(type = "SW-NW")) %>%
    select(-statistic, - method, - alternative, -parameter)
}

appendVar = function(df, dmcVar) {
  # Input: Dataframe of t-test results; DMC variable table
  # Output: Dataframe ot t-test results appended with DMC values
  posNames = colnames(dmcVar %>% select(-site))
  posDf = as.data.frame(matrix(posNames,1,length(posNames)))
  colnames(posDf) = paste0("Pos",1:length(posNames))
  
  nwBases = dmcVar %>% filter(site == "NW") %>% select(-site)
  colnames(nwBases) = paste0("NW_", 1:length(posNames))
  
  seBases = dmcVar %>% filter(site == "SE") %>% select(-site)
  colnames(seBases) = paste0("SE_", 1:length(posNames))
  
  appendDf = bind_cols(posDf, nwBases, seBases) %>% mutate(cross = 1)
  left_join(df %>% mutate(cross = 1), appendDf) %>% select(-cross)
}
####

findSplitNN = function(seqStr) {
  # Input: Character string of single DMC variable
  # Output: Dataframe of base and position 
  
  reStr = gregexpr("[ACGT]", seqStr)
  varStr = paste0("Var", reStr %>% unlist())
  if (varStr[1] == "Var5" & varStr[2] == "Var6") {F} else {T}
}

diMean = function(strVal,df) {
  # Input: strVal = flank of interest
  # Output: dataframe containing mean mononucleotide coefficients
  sites = gregexpr("[A,C,G,T]", strVal) %>% unlist()
  foreach(i = 1:2, .combine = "c")%do%{
    returnVal = "NNNNNNNNNN"
    substring(returnVal,sites[i],sites[i]) <- substring(strVal,sites[i],sites[i])
    df %>% filter(flank == returnVal) %>% .$coef
  } %>% sum()
}

hdset = function(lettDf) {
  # input: lettDf = df of character letter {A,C,G,T} in each column
  # output: 3-row df of 1-HD sets
  lettSet = c("A","C","G","T")
  foreach(i = paste0("X",sprintf("%02d",1:10)), .combine = "rbind")%do%{
    colLett = lettDf[,i] %>% unlist(use.names = F)
    hdVal = lettSet[!(lettSet %in% colLett)]
    hdvalDf = data.frame(V1 = hdVal)
    colnames(hdvalDf) <- i
    returnDf = left_join(hdvalDf %>% mutate(cross = 1), lettDf %>% select_(paste0("-",i)) %>% mutate(cross = 1), by = "cross") %>% select(-cross)
    return(returnDf %>% tbl_df())
  }
}

hdset2 = function(flankSeq,refSeq) {
  # input: flankSeq = vector of flank sequence; refSeq = reference
  # output: vector of hd
  foreach(i = flankSeq, .combine = "c")%do%{
    separatedFlank = i %>% strsplit(.,"") %>% unlist()
    separatedRef = refSeq %>% strsplit(.,"") %>% unlist()
    sum(separatedFlank != separatedRef)
  }
}

recurhd2 = function(refDf, rLevel) {
  # input: refDf = df of assay data that must include flank, X01:X10 and ddG
  # rLevel = int specifying the recursion level
  
  ## determine highest affinity sequence
  bestSlice = refDf %>% filter(ddG == min(ddG))
  
  ## create df to return
  hdDf = bestSlice %>% mutate(cc = 0)
  
  ## trimmed refDf
  trimDf = refDf[!(refDf$flank %in% bestSlice$flank),] # remove repeats across levels
  
  if (nrow(hdDf) < nrow(refDf)) { # add early stopping
    foreach(j = 1:rLevel)%do%{ # j level in concentric circles
      print(paste0("level = ",j))
      jSlice = hdDf %>% filter(cc == (j-1)) # extract sequences from previous level
      
      preDf = foreach(k = 1:nrow(jSlice), .combine = "rbind")%dopar%{ # k = number of elements in previous concentric circle
        kSlice = jSlice %>% slice(k) %>% select(X01:X10)
        returnDf = 
          hdset(kSlice) %>%
          unite(flank, X01:X10, sep = "")
      } %>% 
        distinct() %>% # remove repeats in a single level
        inner_join(.,trimDf) %>%
        mutate(cc = j)
      trimDf = trimDf[!(trimDf$flank %in% preDf$flank),] # update trim df
      hdDf = bind_rows(preDf,hdDf) # update return df
    }
  } else {}
  
  return(hdDf)
}

# Read-in global data
counts.df = 
  fread("~/Google\ Drive/Flanks_data/counts.txt", header = T) %>%
  tbl_df() 
depth = counts.df %>% group_by(protein,rep) %>% summarise(TF = sum(tf_count), IN = sum(ref_count)) %>% ungroup()
counts.df = 
  counts.df %>% 
  na.omit() %>% 
  filter(!is.infinite(ddG)) %>%
  group_by(protein,rep) %>%
  mutate(ddG = ddG-mean(ddG)) %>% # mean-center
  ungroup()

countsNN.df =
  fread("~/Google\ Drive/Flanks_data/all_predicted_ddGs.csv", header = T, sep = ",") %>%
  tbl_df() %>%
  select(flank,Pho4_ddG,Cbf1_ddG) %>% # select ensemble predictions
  gather(key,ddG,-flank) %>% # ddG in RT
  mutate(protein = substring(key,1,4) %>% tolower(), target = flank) %>%
  select(-key) %>%
  separate(target, paste0("X", sprintf("%02d",1:10)), 1:9)

countsNN_scaled = fread("~/Flank_seq/data/scaled_nn_preds.txt", header = T) %>% 
  tbl_df() %>% 
  mutate(protein = ifelse(protein == "Pho4","pho4","cbf1"))

std_kd = bind_rows(fread("~/Google\ Drive/Flanks_data/std_kd_pho4.csv", header = T) %>% 
                     mutate(protein = "pho4") %>%
                     slice(1:31), # remove negative control
                     fread("~/Google\ Drive/Flanks_data/std_kd_cbf1.csv", header = T) %>% 
                     mutate(protein = "cbf1") %>%
                     slice(1:31)) %>%
  tbl_df() %>%
  mutate(flank = paste(substring(Sequence,17,21),substring(Sequence,30,34),sep="")) %>%
  select(flank, estimate, std.error, protein) %>%
  group_by(protein) %>%
  mutate(kd = estimate/1E9,
         stdE.err = (std.error/estimate)*0.593,
         stdE = log(kd) * 0.593,
         target = flank) %>%
  separate(target,paste0("X",sprintf("%02d",1:10),sep=""),1:9) %>%
  ungroup()

# read in sequence data
ref.gen = read.fasta(file = "~/Google\ Drive/Flanks_data/S288C_reference_sequence_R27-1-1_20031001.fsa") #2003

chip.data = 
  tbl_df(read.csv(file = "~/Google\ Drive/Flanks_data/molcel3915mmc2.csv", header = T)) %>%
  filter(Alignability == 1) %>%
  select(chr = CHR, loc = Location, pho4 = PHO4.Enrichment.No.Pi, cbf1 = Cbf1.Enrichemnt.No.Pi) 

chip.gen =
  foreach(i = 1:16, .combine = "rbind")%do%{
    chromo = ref.gen[[i]]
    foreach(j = chip.data %>% filter(chr == i) %>% .$loc, .combine = "rbind")%do%{
      window = toupper(paste(chromo[(j-7):(j+8)], collapse = "")) #10 bp search window
      flank = paste(substring(window,1,5), substring(window,12,17),sep="")
      chip.data %>% filter(chr == i, loc == j) %>% mutate(flank = flank)
    }
  }

chip.gen.full =
  bind_rows(chip.gen %>% mutate(dir = 0), chip.gen %>% group_by(chr,loc,pho4,cbf1) %>% mutate(dir = 1, flank = revcomp(flank))) %>% 
  distinct() %>% 
  select(flank, pho4, cbf1)

# Read in model output data and put into simple model form
model_outputs <- fread("~/Google\ Drive/Flanks_data/model_outputs.txt", header = T)
