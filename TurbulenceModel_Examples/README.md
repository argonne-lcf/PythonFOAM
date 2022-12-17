# Compiling turbulence models that call Python
## Before starting

Turbulence models that call Python, at the moment, require a duplicate solver with arglist initialization as false (this is to allow for compatibility with Numpy). See issue here: 
https://github.com/pybind/pybind11/issues/1889#issuecomment-1029909985

## Steps to deploy a new RANS turbulence model in OpenFOAM 8
1. Go to `TurbulenceModels/momentumTransportModels/RAS` and make a copy of a pre-existing model that is closest to your desired new model. Here we have copied `kEpsilon/` and named it `PykEpsilon/`. Within this folder update all the file and class names from `kEpsilon` to `PykEpsilon`. 

2. See edits in `PykEpsilon.C` and `PykEpsilon.H` as to how Python is now being initialized when this class is constructed and called repeatedly when the turbulence model is evaluated. Here - we have simply performed the same case study as `PODFoam/` except that the online SVD snapshot collection is happening from inside the turbulence model and not the main solver. 

3. Go to `TurbulenceModels/incompressible/kinematicMomentumTransportModels/kinematicMomentumTransportModels.C` and add our newly copied and edited turbulence model (see lines 101-102).

4. From `TurbulenceModels/` run `compile_new_models.sh` so that your new turbulence model is constructed and registered with OpenFOAM.

5. Using your new turbulence model from a pre-existing solver (say simpleFOAM) may cause segfault/floating point errors due to numpy incompatibility. For this reason, we need to make a duplicate solver that has a workaround for this. An example of this solver is provided in `PysimpleFoam` where we have simply copied simpleFoam, changed its name, added the lines 44-49 in `PysimpleFoam.c` and changed the executable/file name in `PysimpleFoam/Make/files`. You can compile this solver from `PysimpleFoam/` using `wclean && wmake`.

6. To test if our `PykEpsilon` model works appropriately go to the `PysimpleFoam/PitzDaily` case and execute it using `PysimpleFoam`. You will note that the `constant/momentumTransport` file has our new model mentioned there and the `system/controlDict` file has our new (duplicate) solver. 


Follow the steps carefully and you will have your Python-OpenFOAM coupling from a turbulence model ---- Great for data-driven turbulence modeling! Reach out to me for more questions. 