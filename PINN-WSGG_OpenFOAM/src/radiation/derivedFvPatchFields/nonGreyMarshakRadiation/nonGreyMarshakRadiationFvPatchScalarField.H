/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
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

Class
    Foam::nonGreyMarshakRadiationFvPatchScalarField

Group
    grpThermoBoundaryConditions

Description
    A 'mixed' boundary condition that implements a Marshak condition for the
    incident radiation field (usually written as G)

    The radiation temperature is retrieved from the mesh database, using a
    user specified temperature field name.

Usage
    \table
        Property     | Description             | Required    | Default value
        T            | temperature field name  | no          | T
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            MarshakRadiation;
        T               T;
        value           uniform 0;
    }
    \endverbatim

See also
    Foam::radiationCoupledBase
    Foam::mixedFvPatchField

SourceFiles
    nonGreyMarshakRadiationFvPatchScalarField.C

\*---------------------------------------------------------------------------*/

#ifndef radiation_nonGreyMarshakRadiationFvPatchScalarField_H
#define radiation_nonGreyMarshakRadiationFvPatchScalarField_H

#include "mixedFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace radiation
{

/*---------------------------------------------------------------------------*\
             Class nonGreyMarshakRadiationFvPatchScalarField Declaration
\*---------------------------------------------------------------------------*/

class nonGreyMarshakRadiationFvPatchScalarField
:
    public mixedFvPatchScalarField
{

    // Private data

        //- Name of temperature field
        word TName_;


public:

    //- Runtime type information
    TypeName("nonGreyMarshakRadiation");


    // Constructors

        //- Construct from patch and internal field
        nonGreyMarshakRadiationFvPatchScalarField
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        nonGreyMarshakRadiationFvPatchScalarField
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given MarshakRadiationFvPatchField onto a new
        //  patch
        nonGreyMarshakRadiationFvPatchScalarField
        (
            const nonGreyMarshakRadiationFvPatchScalarField&,
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        nonGreyMarshakRadiationFvPatchScalarField
        (
            const nonGreyMarshakRadiationFvPatchScalarField&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchScalarField> clone() const
        {
            return tmp<fvPatchScalarField>
            (
                new nonGreyMarshakRadiationFvPatchScalarField(*this)
            );
        }

        //- Construct as copy setting internal field reference
        nonGreyMarshakRadiationFvPatchScalarField
        (
            const nonGreyMarshakRadiationFvPatchScalarField&,
            const DimensionedField<scalar, volMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual tmp<fvPatchScalarField> clone
        (
            const DimensionedField<scalar, volMesh>& iF
        ) const
        {
            return tmp<fvPatchScalarField>
            (
                new nonGreyMarshakRadiationFvPatchScalarField(*this, iF)
            );
        }


    // Member functions

        // Access

            //- Return the temperature field name
            const word& TName() const
            {
                return TName_;
            }

            //- Return reference to the temperature field name to allow
            //  adjustment
            word& TName()
            {
                return TName_;
            }


        // Evaluation functions

            //- Update the coefficients associated with the patch field
            virtual void updateCoeffs();


        // I-O

            //- Write
            virtual void write(Ostream&) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace radiation
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
