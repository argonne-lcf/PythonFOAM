# PythonFOAM:
## In-situ data analyses with OpenFOAM and Python

Using Python modules for in-situ data analytics with [OpenFOAM](https://www.openfoam.com).
**NOTE** that this is _NOT_ PyFOAM which is an automation tool for running OpenFOAM cases. What you see in this repository, is _OpenFOAM calling Python functions and classes_ for in-situ data analytics. You may offload some portion of your compute task to Python for a variety of reasons (chiefly data-driven tasks using the Python ML ecosystem and quick prototyping of algorithms).

OpenFOAM versions that should compile without changes:
- openfoam.com versions: v2012, v2106
- openfoam.org versions: 8


## Prerequisites

- OpenFOAM
- numpy (python) with devel headers
- tensorflow (python)
- matplotlib.pyplot (python)


## Contents
1. `PODFoam/`: A `pimpleFoam` solver with in-situ collection of snapshot data for a streaming singular value decomposition. Python bindings are used to utilize a Python Streaming-SVD class object from OpenFOAM.

2. `APMOSFoam/`: A `pimpleFoam` solver with in-situ collection of snapshot data for a parallelized singular value decomposition. While the previous example performs the SVD on data only on one rank - this solver performs a global, but distributed, SVD. However, SVD updates are not streaming.

## To compile and run

Use standard procedure to compile a new solver in OpenFOAM, i.e., use `wmake` from within `PODFoam/`.

To run cases, it is assumed that you have a Python virtual environment that is linked to during compile and run time.
The relevant lines are
```
PY_INCS
PY_LIBS
```
within `Make/options`.
**Replace these with the include/lib paths to your personal Python environments.** The Python module within `Run_Case/` directories require the use of `numpy`, `matplotlib`, and `tensorflow` so ensure that your environment has these installed. The best way to obtain these is to `pip install tensorflow==2.1` which will automatically find the right numpy dependency and then `pip install matplotlib` to obtain plot capability. You will also need to install `mpi4py` which you can using `pip install mpi4py`.

## Update
A Docker container with the contents of this repo is available [here](https://hub.docker.com/repository/docker/romitmaulik1/pythonfoam_docker). You can use `docker pull romitmaulik1/pythonfoam_docker:reproduced` on a machine with docker in it, or `singularity build pythonfoam.img docker://romitmaulik1/pythonfoam_docker:reproduced`. Do not forget to ensure OpenFOAM is sourced and available in your path by using `source /opt/openfoam8/etc/bashrc`. For a quick crash course on using Docker, see this tutorial by [Jean Rabault](https://github.com/jerabaul29/Cylinder2DFlowControlDRLParallel/blob/master/Docker/README_container.md). Singularity resources may be found [here](https://github.com/argonne-lcf/CompPerfWorkshop-2021/blob/main/03_containers/ALCF_Containers.pdf).

Points of contact for further assistance - Romit Maulik (rmaulik@anl.gov). This work was performed by using the resources of the Argonne Leadership Computing Facility, a U.S. Department of Energy (Office of Science) user facility at Argonne National Laboratory, Lemont, IL, USA. 

## LICENSE

[Argonne open source](LICENSE) for the Python integration
