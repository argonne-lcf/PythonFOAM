blockMesh &> log.blockMesh
decomposePar &> log.decomposePar
mpirun -np 4 --allow-run-as-root APMOSFoam -parallel &> log.APMOSFoam
