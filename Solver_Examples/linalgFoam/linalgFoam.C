/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
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
    linalgFoam

Description
    Solves a simple Laplace equation, e.g. for thermal diffusion in a solid and ports the generated stiffness matrix 
    over to Python.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fvOptions.H"
#include "simpleControl.H"
#include "simpleMatrix.H"

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
    Foam::argList args(argc, argv, true,true,/*initialise=*/false);
    if (!args.checkRootCase())
    {
        Foam::FatalError.exit();
    }

    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    Info<< "Initialize python" << nl << endl;
    Py_Initialize();
    Info<< "Python initialize successful" << nl << endl;

    // initialize numpy array library
    init_numpy();
    Info<< "numpy initialize successful" << nl << endl;

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");
    PyRun_SimpleString("print('Hello world from Python interpreter!')");

    PyObject* pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    
    // Inform OpenFOAM about what function name is and how many arguments it takes
    
    PyObject* my_func = PyObject_GetAttrString(pModule, "adv_diff"); // Only getting the name
    
    PyObject* my_func_args = PyTuple_New(3); // Tuple for arguments to my_func

    // Preparing data for sending to python
    PyObject* array_2d(nullptr);
    PyObject* array_1d(nullptr);
    PyObject* array_1d2(nullptr);
    
    

    // Placeholder to grab data before sending to Python
    int num_cells = mesh.cells().size();
    
    
    Info<< "Specifying functions to call from python_module" << endl;

    // Compile OpenFOAM stiffness matrix and forcing vector data in a double array
    // This part follows https://www.tfd.chalmers.se/~hani/kurser/OS_CFD_2022/lectureNotes/fvMatrix.pdf
    // Utilizing the LDU addressing system used by OpenFOAM:
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nCalculating temperature distribution\n" << endl;

    while (simple.loop(runTime))
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix TEqn
            (
                fvm::ddt(T) - fvm::laplacian(DT, T)
             ==
                fvOptions(T)
            );


            // Getting the addresses
            labelUList lAdd = TEqn.lduAddr().lowerAddr();
            labelUList uAdd = TEqn.lduAddr().upperAddr();
            labelUList oAdd = TEqn.lduAddr().ownerStartAddr();

            scalarField lower  = TEqn.lower();
            scalarField upper  = TEqn.upper();
            scalarField diag   = TEqn.diag();
            scalarField source = TEqn.source();

            simpleMatrix<scalar> CoeffMat(num_cells);
            
            int k = 0;
            
            //Allocating the matrix
            for(label i = 0; i < num_cells; i++)
            {
               CoeffMat.source()[i] = 0.0;
               for(label j = 0; j < num_cells; j++)
               {
                   CoeffMat[i][j] = 0.0;
               }
            }
            // Assigning diagonal coefficients
            for(label i = 0; i < num_cells; ++i)
            {
                CoeffMat[i][i] = diag[i];
                CoeffMat.source()[i] = source[i];
            }
            // Assigning off-diagonal coefficients
    
    
            for(label faceI = 0; faceI < lAdd.size(); ++faceI)
            {
                label l        = lAdd[faceI];
                label u        = uAdd[faceI];
                CoeffMat[l][u] = upper[faceI];
                CoeffMat[u][l] = lower[faceI];
            }

            forAll(T.boundaryField(), patchI)
            {
                   const fvPatch &pp = T.boundaryField()[patchI].
                   patch();
                   forAll(pp, faceI)
                   {
                          label cellI = pp.faceCells()[faceI];
                          CoeffMat[cellI][cellI] += TEqn.internalCoeffs()[patchI][faceI];
               
                          CoeffMat.source()[cellI] +=
                          TEqn.boundaryCoeffs()[patchI][faceI];
                   }
            }
             Info<< "\ncreated stiffness matrix\n" << endl;
            
            // Once we successfully create the matrix, let us transfer to Python:

            double input_vals[num_cells][num_cells];
            double rhs_vec[num_cells];
            double sol_vec[num_cells];
            for(int i = 0; i < num_cells; i++)
            { 
                
                for(int j = 0; j < num_cells; j++)
                {
                    input_vals[i][j] = CoeffMat[i][j];
                    rhs_vec[j] = CoeffMat.source()[j];
                }
            
            }


            Info<< "Created input_vals" << endl;
            fvOptions.constrain(TEqn);
            TEqn.solve();
            fvOptions.correct(T);
            for (int k = 0; k < num_cells; k++)
            {
                sol_vec[k] = T[k];
            }
                // Cast to numpy before sharing reference with Python
                // Tell numpy about array dimensions
            npy_intp dim[] = {num_cells, num_cells};
            npy_intp dim2[] = {num_cells, 1};
            
            npy_intp dim3[] = {num_cells, 1};
            
                // create a new array
            
            array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &input_vals);
            array_1d = PyArray_SimpleNewFromData(1, dim2, NPY_DOUBLE, &rhs_vec);
            array_1d2 = PyArray_SimpleNewFromData(1, dim3, NPY_DOUBLE, &sol_vec);


            Info<< "Created numpy arrays (array_2d and array_1d)" << endl;
            PyTuple_SetItem(my_func_args, 0, array_2d);
            PyTuple_SetItem(my_func_args, 1, array_1d);
            PyTuple_SetItem(my_func_args, 2, array_1d2);
            

            //Call function and get a pointer to PyArrayObject
            PyArrayObject *pValue = reinterpret_cast<PyArrayObject*>(
                           PyObject_CallObject(my_func, my_func_args)
                           );

            Info<< "Set array_2d as first argument of my_func_args tuple, array_1d as second argument and array_1d2 as third " << endl;
            

            // Cast dereference of numpy array to double
            double c_out = *((double*)PyArray_GETPTR1(pValue,0));

           

        }

        #include "write.H"

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
