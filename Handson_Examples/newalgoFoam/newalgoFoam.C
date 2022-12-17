/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2020 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    newalgoFoam (calling python from an OpenFOAM solver)

Description
    Steady-state solver for incompressible, turbulent flow, using the SIMPLE
    algorithm.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "kinematicMomentumTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"


// Following code to link with Python
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>
// This initializes numpy for use in OpenFOAM
void init_numpy() {
  import_array1();
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{

    Info << "Entered main function -- all well" << nl << endl;

    // create argument list - this is to prevent numpy error
    Foam::argList args(argc, argv, true,true,/*initialise=*/false);
    if (!args.checkRootCase())
    {
        Foam::FatalError.exit();
    }

    #include "postProcess.H"

    // #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"

    Info<< "Initialize python" << nl << endl;
    Py_Initialize();
    Info<< "Python initialize successful" << nl << endl;

    // initialize numpy array library
    init_numpy();
    Info<< "numpy initialize successful" << nl << endl;

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");
    PyRun_SimpleString("print('Hello world from Python interpreter!')");

    // Load a Python module from a python script in the working directory
    PyObject* pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (!pModule)
    {
        FatalErrorInFunction
            << "Errors loading python_module (missing imports?)" << nl
            << exit(FatalError);
    }

    // Inform OpenFOAM about what function name is and how many arguments it takes
    Info<< "Specifying functions to call from python_module" << endl;
    PyObject* my_func = PyObject_GetAttrString(pModule, "my_func"); // Only getting the name
    PyObject* my_func_args = PyTuple_New(1); // Tuple for arguments to my_func

    // Preparing data for sending to python
    PyObject* array_2d(nullptr);

    // Placeholder to grab data before sending to Python
    int num_cells = mesh.cells().size();
    double input_vals[num_cells][2];

    // Compile Openfoam field data in a double array
    forAll(U.internalField(), id)
        {
            input_vals[id][0] = U[id].x();
            input_vals[id][1] = U[id].y();
        }

    Info<< "Created input_vals" << endl;

    // Cast to numpy before sharing reference with Python
    // Tell numpy about array dimensions
    npy_intp dim[] = {num_cells, 2};
    // create a new array
    array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &input_vals[0]);

    Info<< "Created numpy array (array_2d)" << endl;

    // Make array first argument
    PyTuple_SetItem(my_func_args, 0, array_2d);
    Info<< "Set array_2d as first argument of my_func_args tuple" << endl;

    //Call function and get a pointer to PyArrayObject
    PyArrayObject *pValue = reinterpret_cast<PyArrayObject*>(
            PyObject_CallObject(my_func, my_func_args)
            );

    // Cast dereference of numpy array to double
    double c_out = *((double*)PyArray_GETPTR1(pValue,0));

    // Verify computation
    printf("The sum of the input_array (returned to C++) is %f \n",c_out);

    // Sum of array from C++
    double sumval = 0.0;
    forAll(U.internalField(), id)
    {
        sumval = sumval + U[id].x() + U[id].y();
    }
    
    printf("The sum of the input_array (calculated in C++) is %f \n",sumval);


    return 0; // Leave us

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (simple.loop(runTime))
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity SIMPLE corrector
        {
            #include "UEqn.H"
            #include "pEqn.H"
        }

        laminarTransport.correct();
        turbulence->correct();

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
