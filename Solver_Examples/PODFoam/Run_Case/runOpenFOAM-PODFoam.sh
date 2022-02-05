#!/bin/bash
#SBATCH --job-name=WMC_0
#SBATCH -A TURBDRL
#SBATCH -p bdwall
#SBATCH --nodes 6
#SBATCH --ntasks-per-node=36
#SBATCH --time=08:00:00

module unload intel
module unload intel-mkl
module unload intel-mpi
module load gcc/7.3.0-xyzezhj
module load openmpi/3.1.3-obi56bx
module load python/3.6.7-7eq7ubs
export LD_LIBRARY_PATH=/home/rmaulik/OF8/TF_C_API/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/gpfs/fs1/home/software/spack-0.10.1/opt/spack/linux-centos7-x86_64/gcc-7.3.0/python-3.6.7-7eq7ubsfsxwib5oi7yk5ek7edv3cr7vt/lib:$LD_LIBRARY_PATH
source /home/rmaulik/OF8/OFPYENV/bin/activate
source /home/rmaulik/OF8/OpenFOAM-8/etc/bashrc

export I_MPI_FABRICS=shm:tmi
# export I_MPI_OFI_PROVIDER=psm2
# export I_MPI_EXTRA_FILESYSTEM=1
# export I_MPI_EXTRA_FILESYSTEM_LIST=gpfs

now=$(date "+%m.%d.%Y-%H.%M.%S")
solverLogFile="log.${solver}-${now}"
mpiexec PODFoam -parallel >> ${solverLogFile}