#!/bin/bash
#SBATCH --job-name=Duck          # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de taches (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=8            # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
##SBATCH --cpus-per-task=4           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=02:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=./logs/DuckDuckGeese.out      # nom du fichier de sortie
#SBATCH --error=./logs/err_Duck.out       # nom du fichier d'erreur (ici commun avec la sortie)
 
# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
module load tensorflow-gpu/py2/1.14 openjdk python/2.7.16

# activate the conda environment
conda activate /gpfslocalsup/pub/anaconda-py2/2019.03/envs/tensorflow-gpu-1.14

export PYTHONPATH=$PYTHONPATH:/gpfslocalsup/pub/anaconda-py2/2019.03/lib/python2.7/site-packages
export PYTHONPATH=$PYTHONPATH:/gpfsdswork/projects/rech/pch/ulz67kb/SMATE_MTS/Baselines/mtsc_nmsu_ijcai2020/src

# echo des commandes lancees
set -x
 
# execution du code
 python fcn_ca_main.py DuckDuckGeese 0
