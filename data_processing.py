import subprocess
from Bio import SeqIO
import numpy
import re
from tqdm import *
from collections import defaultdict

# Set directory variables (example directories)
FORWARD_FASTQ = "/path/to/FORWARD.fq"
REVERSE_FASTQ = "/path/to/REVERSE.fq"
OUT_PREFIX = "output"

# Combine reads using PEAR
## 4 threads, 7 G RAM

# Extract flank sequence and UMI from each read in each fraction
## PHRED >= 30 threshold
## Regex on CACGTG
## Deduplicate
## each functional base position must be PHRED >= 30

# Call PEAR
subprocess.call(["pear", "-y", "7G", "-j", "4", "-f", FORWARD_FASTQ,
                 "-r", REVERSE_FASTQ, "-o",
                 "{}_comboreads".format(OUT_PREFIX)])

fraction = "{}_comboreads.assembled.fastq".format(OUT_PREFIX)

# Create a dictionary to hold all reads
seqDict = defaultdict(set)

# Loop through all reads
for record in tqdm(SeqIO.parse(fraction, "fastq")):

    # Check whether mean read quality > 30
    if numpy.mean(record.letter_annotations["phred_quality"]) >= 30:

        # Wrap in try, some sequences may be truncated and will be skipped
        try:
            # Find important positions
            sequence = str(record.seq)
            pattern = re.search("ATC\w\w\w\w\wCACGTG\w\w\w\w\wCTA", sequence)
            sequence_score = record.letter_annotations["phred_quality"]
            all_scores = sequence_score[pattern.start()+3:pattern.end()-3]
            all_scores.append(sequence_score[pattern.start()-25])
            all_scores.append(sequence_score[pattern.start()-24])
            all_scores.append(sequence_score[pattern.start()-13])
            all_scores.append(sequence_score[pattern.start()+23])
            all_scores.append(sequence_score[pattern.start()+24])
            all_scores.append(sequence_score[pattern.start()+12])

            # Check to make sure quality is high enough
            if not sum([element < 30 for element in all_scores]) > 0:
                fullFlank = sequence[pattern.start()+3:pattern.end()-3]
                flank = fullFlank[0:5] + fullFlank[11:16]
                umi = sequence[pattern.start()-25] + \
                      sequence[pattern.start()-24] + \
                      sequence[pattern.start()-13] + \
                      sequence[pattern.end()+12] + \
                      sequence[pattern.end()+23] + \
                      sequence[pattern.end()+24]
                seqDict[flank].add(umi)

        # Continue on exception
        except Exception as e:
            continue

with open("{}_unique_reads.txt".format(OUT_PREFIX), "a") as f1:
    for seq,umi in seqDict.items():
        umiLen = len(umi)
        printline = "{}\t{}\n".format(seq,umiLen)
        f1.write(printline)
