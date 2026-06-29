/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2020 OpenCFD Ltd.
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

\*---------------------------------------------------------------------------*/

#include "nonGreyMeanAbsorptionEmission.H"
#include "addToRunTimeSelectionTable.H"
#include "unitConversion.H"
#include "extrapolatedCalculatedFvPatchFields.H"
#include "basicSpecieMixture.H"
#include "globalModel.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    namespace radiation
    {
        defineTypeNameAndDebug(nonGreyMeanAbsorptionEmission, 0);

        addToRunTimeSelectionTable
        (
            absorptionEmissionModel,
            nonGreyMeanAbsorptionEmission,
            dictionary
        );
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::radiation::nonGreyMeanAbsorptionEmission::nonGreyMeanAbsorptionEmission
(
    const dictionary& dict,
    const fvMesh& mesh
)
:
    absorptionEmissionModel(dict, mesh),
    coeffsDict_((dict.optionalSubDict(typeName + "Coeffs"))),
    nBands_(coeffsDict_.get<label>("nBands")),
    thermo_(mesh.lookupObject<fluidThermo>(basicThermo::dictName)),
    EhrrCoeff_(coeffsDict_.get<scalar>("EhrrCoeff")),
    spectrumModelName_(coeffsDict_.get<word>("spectrumModelName"))
{
    if (!isA<basicSpecieMixture>(thermo_))
    {
        FatalErrorInFunction
            << "Model requires a multi-component thermo package"
            << abort(FatalError);
    }
    Foam::radiation::globalModel* spectralModel =
        Foam::radiation::globalModel::getInstance
        (
            spectrumModelName_,
            nBands_,
            thermo_.T().mesh()
        );
    (void)spectralModel;
}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::radiation::nonGreyMeanAbsorptionEmission::~nonGreyMeanAbsorptionEmission()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField>
Foam::radiation::nonGreyMeanAbsorptionEmission::aCont(const label bandI) const
{
    Foam::radiation::globalModel* spectralModel =
        Foam::radiation::globalModel::getInstance();
    return spectralModel->get_k(bandI);
}


Foam::tmp<Foam::volScalarField>
Foam::radiation::nonGreyMeanAbsorptionEmission::eCont(const label bandI) const
{
    Foam::radiation::globalModel* spectralModel =
        Foam::radiation::globalModel::getInstance();
    return spectralModel->get_a(bandI);
}


Foam::tmp<Foam::volScalarField>
Foam::radiation::nonGreyMeanAbsorptionEmission::ECont(const label bandI) const
{
    tmp<volScalarField> E
    (
        new volScalarField
        (
            IOobject
            (
                "ECont" + name(bandI),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimensionedScalar(dimMass/dimLength/pow3(dimTime), Zero)
        )
    );

    const volScalarField* QdotPtr = mesh_.findObject<volScalarField>("Qdot");

    if (QdotPtr)
    {
        const volScalarField& Qdot = *QdotPtr;

        if (Qdot.dimensions() == dimEnergy/dimTime)
        {
            E.ref().primitiveFieldRef() = EhrrCoeff_*Qdot/mesh_.V();
        }
        else if (Qdot.dimensions() == dimEnergy/dimTime/dimVolume)
        {
            E.ref().primitiveFieldRef() = EhrrCoeff_*Qdot;
        }
        else
        {
            if (debug)
            {
                WarningInFunction
                    << "Incompatible dimensions for Qdot field" << endl;
            }
        }
    }
    else
    {
        // WarningInFunction
        //   << "Qdot field not found in mesh" << endl;
    }

    return E;
}

void Foam::radiation::nonGreyMeanAbsorptionEmission::correct
    (
        volScalarField& a,
        PtrList<volScalarField>& aj
    ) const
{
    Foam::radiation::globalModel* spectralModel =
        Foam::radiation::globalModel::getInstance();
    spectralModel->updateTref();
    spectralModel->updateKA();
}




// ************************************************************************* //
