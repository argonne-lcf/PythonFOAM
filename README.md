# PythonFOAM:
## In-situ data analyses with OpenFOAM and Python

Using Python modules for in-situ data analytics with [OpenFOAM](https://www.openfoam.com).
**NOTE** that this is _NOT_ PyFOAM which is an automation tool for running OpenFOAM cases. What you see in this repository, is _OpenFOAM calling Python functions and classes_ for in-situ data analytics. You may offload some portion of your compute task to Python for a variety of reasons (chiefly data-driven tasks using the Python ML ecosystem and quick prototyping of algorithms).

OpenFOAM versions that should compile without changes:
- openfoam.com versions: v2012, v2106
- openfoam.org versions: 8

You can find an extensive hands-on tutorial, courtesy of the ALCF PythonFOAM workshop, here: https://www.youtube.com/watch?v=-Sa2OEssru8

## Prerequisites

- OpenFOAM
- numpy (python) with devel headers
- tensorflow (python) ### Version 2.1
- matplotlib (python)
- python-dev-tools (python)

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

## Update - 12/16/2022

You can now debug the C++ components of PythonFOAM with visual studio code. For this you need to have OpenFOAM-8 built in debug mode. Here is a quick tutorial to do so:

1. Download OpenFOAM-8 source
```
git clone https://github.com/OpenFOAM/OpenFOAM-8.git
git clone https://github.com/OpenFOAM/ThirdParty-8.git
```
Go to line 84 in `OpenFOAM-8/etc/bashrc` and 
```
export WM_COMPILE_OPTION=Debug
```
then use `source OpenFOAM-8/etc/bashrc` to load environment variables. After this step, go to `ThirdParty-8/` and use `./Allwmake`. After - go to `OpenFOAM-8/` and use `./Allwmake -j`. (Note we are skipping Paraview compilation). We recommend keeping one build of debug OpenFOAM and one build of optimized OpenFOAM on your system at all times.

2. Download Visual studio and make sure your visual studio has C/C++ (intellisense and extension pack) extensions. 

3. Navigate to your solver build directory - here let us use `PODFoam_Debug/` as an example. This folder has the files and `wmake` instructions to build `PODFoam_Debug` - you will note that the folder also shares the directories required to run a CFD case (i.e., the contents of `run_case/` are in the same build directory). This is required for debug mode execution of our solver. 

4. Create a new hidden folder in the `PODFoam_Debug` directory called `.vscode/`. In it create 4 files
```
launch.json
c_cpp_properties.json
tasks.json
settings.json
```
Use the files in `PODFoam_Debug/.vscode` in this repository to add file contents (further information here: https://github.com/Rvadrabade/Debugging-OpenFOAM-with-Visual-Studio-Code/).

5. In a new terminal - `source prep_env.sh -debug` to ensure that you are running with the debug version of OpenFOAM. Note that here you have to make sure you are pointing to your correct bashrc. The links in this example are for my personal machine. Follow previous steps to compile a debug version of `PODFoam_Debug` from `PODFoam_Debug/`. There should be no issues here.

6. Navigate to `PODFoam_Debug/` and run visual studio code with `code .`. Set a breakpoint in `PODFoam_Debug.C` and hit F5 in the debug panel to initialize debugging. Standard gdb rules apply hereon.

### Note we are still investigating mixed-mode debugging for C++ and Python.


## Docker

A Docker container with the contents of this repo is available [here](https://hub.docker.com/repository/docker/romitmaulik1/pythonfoam_docker). You can use 

```docker pull romitmaulik1/pythonfoam_docker:latest``` 

on a machine with docker in it to download an image that has PythonFOAM set up on it. Subsequently

```
docker run -t -d -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --privileged --name pythonfoam_container romitmaulik1/pythonfoam_docker
xhost +local:docker # For running GUI applications from docker
docker start pythonfoam_container
docker exec -i -t pythonfoam_container /bin/bash
```

will create a container (named `pythonfoam_container`) from the image and start a shell for you to run experiments. Navigate to `/home/PythonFOAM` within the shell to obtain the source code and test cases. For a quick crash course on using Docker, see this tutorial by [Jean Rabault](https://github.com/jerabaul29/Cylinder2DFlowControlDRLParallel/blob/master/Docker/README_container.md). 

Points of contact for further assistance - Romit Maulik (rmaulik@anl.gov). This work was performed by using the resources of the Argonne Leadership Computing Facility, a U.S. Department of Energy (Office of Science) user facility at Argonne National Laboratory, Lemont, IL, USA. Several aspects of this research were also performed at the Department of Computer Science at IIT-Chicago ([SPEAR Team](http://www.cs.iit.edu/~lan/SPEAR-Team.html) with support from NSF Award #[2119294](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2119294&HistoricalAwards=false)-PI Zhiling Lan).

## LICENSE

[Argonne open source](LICENSE) for the Python integration
