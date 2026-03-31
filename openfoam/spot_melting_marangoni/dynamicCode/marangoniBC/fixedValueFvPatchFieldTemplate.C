/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019-2021 OpenCFD Ltd.
    Copyright (C) YEAR AUTHOR, AFFILIATION
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

#include "fixedValueFvPatchFieldTemplate.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
#include "PatchFunction1.H"

//{{{ begin codeInclude
#line 34 "/home/yzk/OpenFOAM/spot_melting_marangoni/0/U/boundaryField/top"
#include "fvcGrad.H"
//}}} end codeInclude


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

// dynamicCode:
// SHA1 = f6cea5d472d7427b897a262d9f5c538bb5015255
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void marangoniBC_f6cea5d472d7427b897a262d9f5c538bb5015255(bool load)
{
    if (load)
    {
        // Code that can be explicitly executed after loading
    }
    else
    {
        // Code that can be explicitly executed before unloading
    }
}

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

makeRemovablePatchTypeField
(
    fvPatchVectorField,
    marangoniBCFixedValueFvPatchVectorField
);

} // End namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::
marangoniBCFixedValueFvPatchVectorField::
marangoniBCFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    parent_bctype(p, iF)
{
    if (false)
    {
        printMessage("Construct marangoniBC : patch/DimensionedField");
    }
}


Foam::
marangoniBCFixedValueFvPatchVectorField::
marangoniBCFixedValueFvPatchVectorField
(
    const marangoniBCFixedValueFvPatchVectorField& rhs,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    parent_bctype(rhs, p, iF, mapper)
{
    if (false)
    {
        printMessage("Construct marangoniBC : patch/DimensionedField/mapper");
    }
}


Foam::
marangoniBCFixedValueFvPatchVectorField::
marangoniBCFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    parent_bctype(p, iF, dict)
{
    if (false)
    {
        printMessage("Construct marangoniBC : patch/dictionary");
    }
}


Foam::
marangoniBCFixedValueFvPatchVectorField::
marangoniBCFixedValueFvPatchVectorField
(
    const marangoniBCFixedValueFvPatchVectorField& rhs
)
:
    parent_bctype(rhs),
    dictionaryContent(rhs)
{
    if (false)
    {
        printMessage("Copy construct marangoniBC");
    }
}


Foam::
marangoniBCFixedValueFvPatchVectorField::
marangoniBCFixedValueFvPatchVectorField
(
    const marangoniBCFixedValueFvPatchVectorField& rhs,
    const DimensionedField<vector, volMesh>& iF
)
:
    parent_bctype(rhs, iF)
{
    if (false)
    {
        printMessage("Construct marangoniBC : copy/DimensionedField");
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::
marangoniBCFixedValueFvPatchVectorField::
~marangoniBCFixedValueFvPatchVectorField()
{
    if (false)
    {
        printMessage("Destroy marangoniBC");
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void
Foam::
marangoniBCFixedValueFvPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    if (false)
    {
        printMessage("updateCoeffs marangoniBC");
    }

//{{{ begin code
    #line 45 "/home/yzk/OpenFOAM/spot_melting_marangoni/0/U/boundaryField/top"
// === Marangoni parameters ===
            const scalar dgdT  = 1.0e-4;   // dγ/dT [N/(m·K)] — positive for inward flow
            const scalar mu_v  = 0.005;     // dynamic viscosity [Pa·s]
            const scalar Tsol  = 1650.0;
            const scalar Tliq  = 1700.0;

            // Temperature field & its gradient
            const auto& T = this->db().lookupObject<volScalarField>("T");
            const tmp<volVectorField> tgradT = fvc::grad(T);
            const volVectorField& gradT = tgradT();

            // Patch geometry
            const vectorField nf(this->patch().nf());           // outward unit normal
            const scalarField delta(1.0 / this->patch().deltaCoeffs()); // cell→face dist

            // Internal cell velocities
            const vectorField Ui(this->patchInternalField());

            vectorField result(this->size(), vector::zero);

            forAll(result, fi)
            {
                const label ci = this->patch().faceCells()[fi];

                // Liquid fraction at this cell
                scalar fl = 0;
                if (T[ci] > Tsol)
                {
                    fl = (T[ci] >= Tliq) ? 1.0 : (T[ci] - Tsol) / (Tliq - Tsol);
                }

                if (fl < 1e-6)
                {
                    // Solid — zero velocity at surface
                    result[fi] = vector::zero;
                    continue;
                }

                // Tangential temperature gradient
                const vector gT = gradT[ci];
                const vector gT_t = gT - (gT & nf[fi]) * nf[fi];

                // Tangential internal velocity
                const vector Ui_t = Ui[fi] - (Ui[fi] & nf[fi]) * nf[fi];

                // Marangoni: u_face = u_int + (dγ/dT / μ) · ∇_sT · δ
                result[fi] = fl * (Ui_t + (dgdT / mu_v) * gT_t * delta[fi]);
            }

            this->operator==(result);
//}}} end code

    this->parent_bctype::updateCoeffs();
}


// ************************************************************************* //

