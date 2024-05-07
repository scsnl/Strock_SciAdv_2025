# ------------------
# PROJECT VARS
# ------------------
export PROJECT_NAME=2021_md_excitability_nn_model_redo_clean
export PROJECT_LEAD=astrock

# ------------------
# PROJECT PATHS
# ------------------
export PROJECT_OAK=$OAK/projects/$PROJECT_LEAD/$PROJECT_NAME
export PROJECT_GROUP_SCRATCH=$GROUP_SCRATCH/projects/$PROJECT_LEAD/$PROJECT_NAME
export PROJECT_SCRATCH=$SCRATCH/$PROJECT_NAME
export FIG_PATH=${PROJECT_SCRATCH}/figures
export DATA_PATH=${PROJECT_SCRATCH}/data
export TMP_PATH=${PROJECT_SCRATCH}/data
export CODE_PATH=$OAK/projects/$PROJECT_LEAD/Strock_bioRxiv_2024
mkdir -p $TMP_PATH/log

# Change the following variables to adapt to your own platform
#export PROJECT_PATH=.
#export CODE_PATH=${PROJECT_PATH}
#export FIG_PATH=${PROJECT_PATH}/figures
#export DATA_PATH=${PROJECT_PATH}/data

# ------------------
# PYTHON SETUP
# ------------------
ml system libnvidia-container
source $GROUP_HOME/python_envs/pyenv/activate.sh
export NN_COMMON=$CODE_PATH/common
export PYTHONPATH=$NN_COMMON:$PYTHONPATH
source $NN_COMMON/slurm/slurm_aliases.sh
