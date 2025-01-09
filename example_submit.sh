#!/bin/bash

#Indicate the account to charge the job to
#SBATCH --account=mdatascience_team

#Indicate a name to give the job. This can be anything, it's just a way to be able to easily track different jobs
#SBATCH --job-name=example-submit

#Indicate where to send information about the job
#SBATCH --mail-user=<uniqname>@umich.edu

#Indicate how often you want info about the job. In this case, you will receive an email when the job has ended
#SBATCH --mail-type=END

#Indicate how many nodes to run the job on
#SBATCH --nodes=1

#Indicate how many tasks to run on each node
#SBATCH --ntasks-per-node=1

#Indicate how many cpus to use per task
#SBATCH --cpus-per-task=1

#Indicate how much memory to use per cpu
#SBATCH --mem-per-cpu=10000m

#Indicate how long to run the job for
#SBATCH --time=1:00:00

#Indicate which partition to run the job on. In this case, we will run on the standard partition
#SBATCH --partition=standard

#Get rid of any modules that might still be running
module purge

# Create virtual env
python3 -m venv myenv
source ...

# Install the required packages from requirements.txt
pip3 ...

# Run the desired program
python3 <py_filename>
