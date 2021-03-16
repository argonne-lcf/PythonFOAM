#include "createRDeltaT.H"

Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

// Datastructures for APMOSFoam - will be overwritten
volScalarField upod_ = U.component(vector::X);
volScalarField vpod_ = U.component(vector::Y);
volScalarField wpod_ = U.component(vector::Z);


#include "createPhi.H"


label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, pimple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());


singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::momentumTransportModel> turbulence
(
    incompressible::momentumTransportModel::New(U, phi, laminarTransport)
);

/*The following stuff is for Python interoperability*/
PyObject *array_2d, *rank_val;
PyArrayObject *pValue;
int rank, truncation;

Py_Initialize();
PyRun_SimpleString("import sys");
PyRun_SimpleString("sys.path.append(\".\")");

// initialize numpy array library
import_array1(-1);

PyObject* pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
PyObject* pModule = PyImport_Import(pName);
Py_DECREF(pName);

PyObject* snapshot_func = PyObject_GetAttrString(pModule, "snapshot_func");
PyObject* snapshot_args = PyTuple_New(2);

PyObject* apmos_func = PyObject_GetAttrString(pModule, "apmos_func");
PyObject* apmos_args = PyTuple_New(1);
Py_DECREF(pModule);

// Get process id
rank = Pstream::myProcNo();

// Placeholder to grab data before sending to Python
int num_cells = mesh.cells().size();
double input_vals[num_cells][3];

std::cout << "/* The number of cells on this rank is */" << num_cells << std::endl;

// Number of POD modes
truncation = 5;
/*Done importing Python functionality*/

#include "createMRF.H"
#include "createFvOptions.H"