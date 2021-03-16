# PythonFOAM: 
# In-situ data analyses with OpenFOAM and Python

Using Python modules for in-situ data analytics with OpenFOAM 8. **NOTE** that this is _NOT_ PyFOAM which is an automation tool for running OpenFOAM cases. What you see in this repository, is _OpenFOAM calling Python functions and classes_ for in-situ data analytics. You may offload some portion of your compute task to Python for a variety of reasons (chiefly data-driven tasks using the Python ML ecosystem and quick prototyping of algorithms).

## Contents
1. `PODFoam/`: A `pimpleFoam` solver with in-situ collection of snapshot data for a streaming singular value decomposition. Python bindings are used to utilize a Python Streaming-SVD class object from OpenFOAM.

2. `APMOSFoam/`: A `pimpleFoam` solver with in-situ collection of snapshot data for a parallelized singular value decomposition. While the previous example performs the SVD on data only on one rank - this solver performs a global, but distributed, SVD. However, SVD updates are not streaming.

## To compile and run

Use standard procedure to compile a new solver in OpenFOAM, i.e., use `wmake` from within `PODFoam/`. To run cases, it is assumed that you have a Python virtual environment that is linked to during compile and run time. The relevant lines are 
```
-I/gpfs/fs1/home/rmaulik/OF8/OFPYENV/include/python3.6m/ \
-I/gpfs/fs1/home/rmaulik/OF8/OFPYENV/lib/python3.6/site-packages/numpy/core/include \
```
within `EXE_INC` and 
```
-L/gpfs/fs1/home/software/spack-0.10.1/opt/spack/linux-centos7-x86_64/gcc-7.3.0/python-3.6.7-7eq7ubsfsxwib5oi7yk5ek7edv3cr7vt/lib \
-lpython3.6m
```
within `EXE_LIBS`. **Replace these with the include/lib paths to your personal Python environments.** The Python module within `Run_Case/` directories require the use of `numpy`, `matplotlib`, and `tensorflow` so ensure that your environment has these installed. The best way to obtain these is to `pip install tensorflow==2.1` which will automatically find the right numpy dependency and then `pip install matplotlib` to obtain plot capability. You will also need to install `mpi4py` which you can using `pip install mpi4py`.

Points of contact for further assistance - Romit Maulik (rmaulik@anl.gov). This work was performed by using the resources of the Argonne Leadership Computing Facility, a U.S. Department of Energy (Office of Science) user facility at Argonne National Laboratory, Lemont, IL, USA. 

## LICENSE

[MIT](LICENSE)
