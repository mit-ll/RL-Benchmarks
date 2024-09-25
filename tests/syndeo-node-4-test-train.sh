#!/bin/bash

##########################################################################################
#
#         USAGE:  sbatch <file>.sh
#
#   DESCRIPTION:  Runs container on multi-node GPU cluster.  This must be run using
#                 [Syndeo](https://github.com/mit-ll/Syndeo)
#
#       OPTIONS:  --exclusive:          sets hardware to be exclusive to this job
#                 --job-name:           name of the job
#                 --output:             output file name
#                 --cpus-per-task:      number of cpus set per task
#                 --nodes:              number of nodes to assign to this job
#                 --ntasks:             number of parallel tasks allowed
#                                       (should match --nodes)
#                 --ntasks-per-node:    number of tasks to assign per node
#                 --time:               maximum time before killing job
#                                       "days-hours:min:secs"
#                 --constraint:         type of hardware to use
#                 --partition:          type of partition used
#                 --gres:               gpu resource request
#
#        AUTHOR:  William Li, william.li@ll.mit.edu
#       COMPANY:  MIT Lincoln Laboratory
#       VERSION:  1.0
#       CREATED:  08/07/2023
##########################################################################################

#SBATCH --exclusive
#SBATCH --job-name rl_benchmarks_train
#SBATCH --output logs/rl_benchmarks-gaia-node-4-%j.log
#SBATCH --cpus-per-task=40
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --time 0-02:00:00
#SBATCH --partition=gaia
#SBATCH --distribution=nopack

##########################################################################################
# User Config
##########################################################################################
# Host Config
HOST_WORKING_DIR="/state/partition1/user/$USER"
HOST_RAY_TMPDIR="$HOST_WORKING_DIR/tmp"
HOST_SAVE="$HOME/projects/rl_benchmarks/save/train"

# Singualrity Config
export SINGULARITY_TMPDIR="$HOST_RAY_TMPDIR"    # directory used by Apptainer/Singularity

# Container Config
CONTAINER_SRC_PATH="containers/rl_benchmarks.sif"
CONTAINER_TGT_PATH="$HOST_WORKING_DIR/ray_container.sif"
CONTAINER_RAY_TMP="/domi/ray_tmp"
CONTAINER_SAVE="/domi/rl_benchmarks/save/train"

source src/scripts/setup_ray_head.sh \
    --tmpdir=$HOST_RAY_TMPDIR \
    --container_src=$CONTAINER_SRC_PATH \
    --container_tgt=$CONTAINER_TGT_PATH \
    --container_ray_tmp=$CONTAINER_RAY_TMP

source src/scripts/setup_ray_workers.sh \
    --tmpdir=$HOST_RAY_TMPDIR \
    --gpus=0 \
    --index=1 \
    --container_src=$CONTAINER_SRC_PATH \
    --container_tgt=$CONTAINER_TGT_PATH

##########################################################################################
# Container Run
##########################################################################################
# --userns:             run container in new namespace allowing unpriviledged use
# --no-home:            disable mounting the home directory
# --writable-tmpfs:     enable write access within container, but do not save on exit
# --bind:               bind host system paths to container paths

PYTHON_SCRIPT="tests/train_test.py"

singularity run \
    --tmp-sandbox \
    --writable \
    --nv \
    --bind $RAY_TMPDIR:$CONTAINER_RAY_TMP,$HOST_SAVE:$CONTAINER_SAVE \
    $CONTAINER_TGT_PATH \
    $PYTHON_SCRIPT $HEAD_NODE_ADDR $RAY_TMPDIR

##########################################################################################
# Cleanup
##########################################################################################
source src/scripts/shutdown_ray.sh
