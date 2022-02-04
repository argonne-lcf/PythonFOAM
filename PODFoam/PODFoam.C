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
    PODFoam

Description
    Transient solver for incompressible, turbulent flow of Newtonian fluids,
    with optional mesh motion and mesh topology changes.

    Turbulence modelling is generic, i.e. laminar, RAS or LES may be selected.

    Added ability to send data to a python module for performing in-situ POD

\*---------------------------------------------------------------------------*/

// Use great instead of GREAT etc
#define COMPAT_OPENFOAM_ORG
#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "singlePhaseTransportModel.H"
#ifdef OPENFOAM
    #include "turbulentTransportModel.H"
#else
    #include "kinematicMomentumTransportModel.H"
#endif
#include "pimpleControl.H"
#include "CorrectPhi.H"
#include "fvOptions.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"

// Some time related libraries
#include <ctime>

/*The following stuff is for Python interoperability*/
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL POD_ARRAY_API
#include <numpy/arrayobject.h>

void init_numpy() {
  import_array1();
}
/*Done importing Python functionality*/

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{


    // create argument list
    Foam::argList args(argc, argv, true,true,/*initialise=*/false);
    if (!args.checkRootCase())
    {
        Foam::FatalError.exit();
    }

    // Some time related variables
    struct timespec tw1, tw2;
    double posix_wall;

    #include "postProcess.H"
    // #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "initContinuityErrs.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "createUfIfPresent.H"

    #include "PythonCreate.H"

    turbulence->validate();

    if (!LTS)
    {
        #include "CourantNo.H"
        #include "setInitialDeltaT.H"
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    #if OPENFOAM
    while (runTime.run())
    #else
    while (pimple.run(runTime))
    #endif
    {
        #include "readDyMControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "CourantNo.H"
            #include "setDeltaT.H"
        }

        runTime++;

        // Info<< "Start Time = " << runTime.timeName() << nl << endl;
        clock_gettime(CLOCK_MONOTONIC, &tw1); // POSIX

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            if
            (
                #if OPENFOAM
                (pimple.firstIter() || moveMeshOuterCorrectors)
                #else
                (pimple.firstPimpleIter() || moveMeshOuterCorrectors)
                #endif
            )
            {
                mesh.update();

                if (mesh.changing())
                {
                    MRF.update();

                    if (correctPhi)
                    {
                        // Calculate absolute flux
                        // from the mapped surface velocity
                        phi = mesh.Sf() & Uf();

                        #include "correctPhi.H"

                        // Make the flux relative to the mesh motion
                        fvc::makeRelative(phi, U);
                    }

                    if (checkMeshCourantNo)
                    {
                        #include "meshCourantNo.H"
                    }
                }
            }

            #include "UEqn.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                laminarTransport.correct();
                turbulence->correct();
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &tw2); // POSIX
        posix_wall = 1000.0*tw2.tv_sec + 1e-6*tw2.tv_nsec - (1000.0*tw1.tv_sec + 1e-6*tw1.tv_nsec);
        printf("PDE Compute wall time: %.2f ms\n", posix_wall);

        // Info<< "Solver Elapsed ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        //     << "Solver Elapsed ClockTime = " << runTime.elapsedClockTime() << " s"
        //     << nl << endl;

        // Talk to Python
        #include "PythonComm.H"

        // Info<< "Python Elapsed ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        //     << "Python Elapsed ClockTime = " << runTime.elapsedClockTime() << " s"
        //     << nl << endl;

        // Perform IO
        clock_gettime(CLOCK_MONOTONIC, &tw1); // POSIX

        runTime.write();

        clock_gettime(CLOCK_MONOTONIC, &tw2); //POSIX
        posix_wall = 1000.0*tw2.tv_sec + 1e-6*tw2.tv_nsec - (1000.0*tw1.tv_sec + 1e-6*tw1.tv_nsec);
        printf("IO wall time: %.2f ms\n", posix_wall);

        // Info<< "IO Elapsed ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        //     << "IO Elapsed ClockTime = " << runTime.elapsedClockTime() << " s"
        //     << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
