#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=48:00:00

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=4G

# Specify a job name:
#SBATCH -J MyJob1

# Specify an output file
#SBATCH -o IRJob1.out
#SBATCH -e IRJob2.out

# Set up the environment by loading modules
module load java/jdk-17.0.2

# Run a script
java -jar Build_Pubmed_Index.jar -cp LuceneJARFiles2

