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



// if (PyErr_Occurred()) {
//     Info<< "Failed to import numpy Python module(s)." << endl;
//     return NULL; // Or some suitable return value to indicate failure.
// }
// assert(PyArray_API);



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

Info<< "    get svd_func" << endl;
PyObject *svd_func = PyObject_GetAttrString(pModule, "svd_func");
PyObject *svd_args = PyTuple_New(1);

Py_DECREF(pModule);
Info<< "..." << endl;


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Datastructures - will be overwritten
volScalarField upod_(U.component(vector::X));
volScalarField vpod_(U.component(vector::Y));
volScalarField wpod_(U.component(vector::Z));

// To pass rank to Python interpreter
int rank = Pstream::myProcNo();
PyObject *rank_val = PyLong_FromLong(rank);
PyObject *array_2d(nullptr);

///int encode_mode = 0;

// Number of POD modes
int truncation = 5;

// Placeholder to grab data before sending to Python
int num_cells = mesh.cells().size();
auto input_vals = new double[num_cells][3];

// ************************************************************************* //
