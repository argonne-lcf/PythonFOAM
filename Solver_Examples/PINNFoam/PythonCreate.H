// Copyright (C) 2021 UChicago Argonne, LLC
// SPDX-License-Identifier: (other)
//
// Python-OpenFOAM integration
//
// Initialize python, load module and create some scratch space etc.

Info<< "initialize python" << endl;
Py_Initialize();
Info<< "python initialize successful" << endl;

// initialize numpy array library
init_numpy();
Info<< "numpy initialize successful" << endl;

PyRun_SimpleString("import sys");
PyRun_SimpleString("sys.path.append(\".\")");


if (PyErr_Occurred()) {
    Info<< "Failed to import numpy Python module(s)." << endl;
    return NULL; // Or some suitable return value to indicate failure.
}
assert(PyArray_API);


PyObject* pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
PyObject* pModule = PyImport_Import(pName);
Py_DECREF(pName);

if (!pModule)
{
    FatalErrorInFunction
        << "Errors loading python_module (missing imports?)" << nl
        << exit(FatalError);
}

Info<< "    get snapshot_func" << endl;
PyObject* snapshot_func = PyObject_GetAttrString(pModule, "snapshot_func");
PyObject* snapshot_args = PyTuple_New(2);

Info<< "    get pinn_train_func" << endl;
PyObject *pinn_train_func = PyObject_GetAttrString(pModule, "pinn_train_func");
PyObject *pinn_train_args = PyTuple_New(1);


Info<< "    get pinn_deploy_func" << endl;
PyObject *pinn_deploy_func = PyObject_GetAttrString(pModule, "pinn_deploy_func");
PyObject *pinn_deploy_args = PyTuple_New(2);

Py_DECREF(pModule);
Info<< "..." << endl;


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// To pass rank to Python interpreter
int rank = Pstream::myProcNo();
PyObject *rank_val = PyLong_FromLong(rank);
PyObject *array_2d(nullptr);

// Placeholder to grab training data before sending to Python
int num_cells = mesh.cells().size();
auto training_data = new double[num_cells][6];
auto input_data = new double[num_cells][2];

// coordinate data
volScalarField cx_ = mesh.C().component(vector::X);
volScalarField cy_ = mesh.C().component(vector::Y);

// Hyperparameters for training and testing
int pinn_iter = 0;
int deploy_iter = 0;
int num_train_steps = 100;
int num_deploy_steps = 100;


// ************************************************************************* //
