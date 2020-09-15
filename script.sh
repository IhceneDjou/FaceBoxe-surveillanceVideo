#!/bin/bash

#SBATCH -J FaceDetection_Job    
             # Job name
#SBATCH -o FaceDetection_.%j.out  
             # Name of std out output file (%j expands to jobId)
#SBATCH -N 1                   
             # Total number of nodes requested
##SBATCH -n 1                   
             # Number of tasks per node (default = 1)
#SBATCH -p gpu      
             # Total number of mpi tasks requested
#SBATCH --nodelist=node11
             # Total number of mpi tasks requested
#SBATCH --gpu 2             
             # Total number of mpi tasks requested
#SBATCH -t 06:00:00          
             # Run time (hh:mm:ss) - 6hours

# Launch

./make.sh
python train.py
             # in case of training

