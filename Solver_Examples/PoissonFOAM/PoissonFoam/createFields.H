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
//////////////////////////////////////////////////
volScalarField pp
(
IOobject
(
"pp",
runTime.timeName(),
mesh,
IOobject::NO_READ,
IOobject::AUTO_WRITE
),p
);

///////////////////////////////////////////////////
/*volScalarField loss
(
IOobject
(
"loss",
runTime.timeName(),
mesh,
IOobject::READ_IF_PRESENT,
IOobject::AUTO_WRITE
),mesh
);*/

///////////////////////////////////////////////////
volScalarField press
(
IOobject
(
"press",
runTime.timeName(),
mesh,
IOobject::NO_READ,
IOobject::AUTO_WRITE
),p
);

///////////////////////////////////////////////////

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

surfaceScalarField mDot
(
IOobject
(
"mDot",
runTime.timeName(),
mesh,
IOobject::READ_IF_PRESENT,
IOobject::AUTO_WRITE
),
linearInterpolate(U) & mesh.Sf()
);
#include "createPhi.H"

surfaceVectorField U_avg_f
(
IOobject
(
"U_avg_f",
runTime.timeName(),
mesh,
IOobject::READ_IF_PRESENT,
IOobject::AUTO_WRITE
),
linearInterpolate(U)
);

label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, simple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());


singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::momentumTransportModel> turbulence
(
    incompressible::momentumTransportModel::New(U, phi, laminarTransport)
);

#include "createMRF.H"
#include "createFvOptions.H"
