// Copyright (C) 2021 UChicago Argonne, LLC
// SPDX-License-Identifier: (other)
//
// Python-OpenFOAM integration
//
// Initialize python, load module and create some scratch space etc.

Info<< "initialize python" << endl;
Py_Initialize();
PyRun_SimpleString("import sys");
PyRun_SimpleString("sys.path.append(\".\")");

// initialize numpy array library
import_array1(-1);

Info<< "    load python_module" << endl;
PyObject *pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
PyObject *pModule = PyImport_Import(pName);
Py_DECREF(pName);

if (!pModule)
{
    FatalErrorInFunction
        << "Errors loading python_module (missing imports?)" << nl
        << exit(FatalError);
}

Info<< "    get snapshot_func" << endl;
PyObject *snapshot_func = PyObject_GetAttrString(pModule, "snapshot_func");
PyObject *snapshot_args = PyTuple_New(2);

Info<< "    get autoencoder_func" << endl;
PyObject *autoencoder_func = PyObject_GetAttrString(pModule, "autoencoder_func");
PyObject *autoencoder_args = PyTuple_New(1);

Info<< "    get encode_func" << endl;
PyObject *encode_func = PyObject_GetAttrString(pModule, "encode_func");
PyObject *encode_args = PyTuple_New(2);

Py_DECREF(pModule);
Info<< "..." << endl;


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Datastructures - will be overwritten
volScalarField upod_(U.component(vector::X));
volScalarField vpod_(U.component(vector::Y));
volScalarField wpod_(U.component(vector::Z));

// AE Reconstructed
volScalarField urec_(U.component(vector::X));

// To pass rank to Python interpreter
int rank = Pstream::myProcNo();
PyObject *rank_val = PyLong_FromLong(rank);
PyObject *array_2d(nullptr);

int encode_mode = 0;

// Number of latent space coordinates
// int latent_dim = 4;

// Placeholder to grab data before sending to Python
int num_cells = mesh.cells().size();
auto input_vals = new double[num_cells][1];

// ************************************************************************* //
