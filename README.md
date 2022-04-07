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


## Update - 02/04/2022
We have changed instructions to compile and run our examples by automating some of the environment variable declarations. We have also added an example of calling Python from a turbulence model implementation. 


## Contents
1. `Solver_Examples/`
	1. `PODFoam/`: A `pimpleFoam` solver with in-situ collection of snapshot data for a streaming singular value decomposition. Python bindings are used to utilize a Python Streaming-SVD class object from OpenFOAM.

	2. `APMOSFoam/`: A `pimpleFoam` solver with in-situ collection of snapshot data for a parallelized singular value decomposition. While the previous example performs the SVD on data only on one rank - this solver performs a global, but distributed, SVD. However, SVD updates are not streaming.

	3. `AEFoam/`: A `pimpleFoam` solver with in-situ collection of snapshot data for training a deep learning autoencoder. 

2. `Turbulence_Model_Examples/` (Work in progress)
	See detailed `README.md` in this folder.

## To compile and run

Inspect `prep_env.sh` to set paths to various Python, numpy headers and libraries and to source your OpenFOAM 8 installation. **Replace these with the include/lib paths to your personal Python environments.** The Python module within `Run_Case/` directories of different `Solvers/` require the use of `numpy`, `matplotlib`, and `tensorflow` so ensure that your environment has these installed. The best way to obtain these is to `pip install tensorflow==2.1` which will automatically find the right numpy dependency and then `pip install matplotlib` to obtain plot capability. You will also need to install `mpi4py` which you can using `pip install mpi4py`.

1. Solvers: After running `source prep_env.sh`, to run the solver examples go into the respective folder (for example `PODFoam/`) and use `wclean && wmake` to build your model. Run your solver example from `Run_Case/`. Note the presence of `python_module.py` within `Run_Case/`.

2. Turbulence model examples: See `README.md` in `Turbulence_Model_Examples/`.


## Docker

A Docker container with the contents of this repo is available [here](https://hub.docker.com/repository/docker/romitmaulik1/pythonfoam_docker). You can use 

```docker pull romitmaulik1/pythonfoam_docker:latest``` 

on a machine with docker in it to download an image that has PythonFOAM set up on it. Subsequently

```
docker run -t -d -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --name pythonfoam_container romitmaulik1/pythonfoam_docker
xhost +local:docker # For running GUI applications from docker
docker start pythonfoam_container
docker exec -i -t pythonfoam_container /bin/bash
```

will create a container (named `pythonfoam_container`) from the image and start a shell for you to run experiments. Navigate to `/home/PythonFOAM` within the shell to obtain the source code and test cases. For a quick crash course on using Docker, see this tutorial by [Jean Rabault](https://github.com/jerabaul29/Cylinder2DFlowControlDRLParallel/blob/master/Docker/README_container.md). 

Points of contact for further assistance - Romit Maulik (rmaulik@anl.gov). This work was performed by using the resources of the Argonne Leadership Computing Facility, a U.S. Department of Energy (Office of Science) user facility at Argonne National Laboratory, Lemont, IL, USA. 

## LICENSE

[Argonne open source](LICENSE) for the Python integration
