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

#include "codedFvOptionTemplate.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
#include "fvMatrix.H"

//{{{ begin codeInclude

//}}} end codeInclude


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace fv
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

// dynamicCode:
// SHA1 = 42a2905ff649dd172de70073d29cea588b255e47
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void mushyZoneDarcy_42a2905ff649dd172de70073d29cea588b255e47(bool load)
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

defineTypeNameAndDebug(mushyZoneDarcyFvOptionvectorSource, 0);
addRemovableToRunTimeSelectionTable
(
    option,
    mushyZoneDarcyFvOptionvectorSource,
    dictionary
);

} // End namespace fv
} // End namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::fv::
mushyZoneDarcyFvOptionvectorSource::
mushyZoneDarcyFvOptionvectorSource
(
    const word& name,
    const word& modelType,
    const dictionary& dict,
    const fvMesh& mesh
)
:
    fv::cellSetOption(name, modelType, dict, mesh)
{
    if (false)
    {
        printMessage("Construct mushyZoneDarcy fvOption from dictionary");
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::fv::
mushyZoneDarcyFvOptionvectorSource::
~mushyZoneDarcyFvOptionvectorSource()
{
    if (false)
    {
        printMessage("Destroy mushyZoneDarcy");
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void
Foam::fv::
mushyZoneDarcyFvOptionvectorSource::correct
(
    GeometricField<vector, fvPatchField, volMesh>& fld
)
{
    if (false)
    {
        Info<< "mushyZoneDarcyFvOptionvectorSource::correct()\n";
    }

//{{{ begin code
    
//}}} end code
}


void
Foam::fv::
mushyZoneDarcyFvOptionvectorSource::addSup
(
    fvMatrix<vector>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "mushyZoneDarcyFvOptionvectorSource::addSup()\n";
    }

//{{{ begin code - warn/fatal if not implemented?
    #line 165 "/home/yzk/OpenFOAM/spot_melting_marangoni/constant/fvOptions/darcyDamping"
const scalar Cu=1e6, eps=1e-3, Tsol=1650, Tliq=1700, rhoRef=7900;
        const volScalarField& T=mesh().lookupObject<volScalarField>("T");
        const scalarField& V=mesh().V();
        forAll(eqn.diag(),i){
            scalar fl;
            if(T[i]<=Tsol) fl=0;
            else if(T[i]>=Tliq) fl=1;
            else fl=(T[i]-Tsol)/(Tliq-Tsol);
            scalar S=-Cu*sqr(1.0-fl)/(pow3(fl)+eps);
            eqn.diag()[i]+=S*rhoRef*V[i];
        }
//}}} end code
}


void
Foam::fv::
mushyZoneDarcyFvOptionvectorSource::addSup
(
    const volScalarField& rho,
    fvMatrix<vector>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "mushyZoneDarcyFvOptionvectorSource::addSup(rho)\n";
    }

//{{{ begin code - warn/fatal if not implemented?
    #line 180 "/home/yzk/OpenFOAM/spot_melting_marangoni/constant/fvOptions/darcyDamping"
const scalar Cu=1e6, eps=1e-3, Tsol=1650, Tliq=1700, rhoRef=7900;
        const volScalarField& T=mesh().lookupObject<volScalarField>("T");
        const scalarField& V=mesh().V();
        forAll(eqn.diag(),i){
            scalar fl;
            if(T[i]<=Tsol) fl=0;
            else if(T[i]>=Tliq) fl=1;
            else fl=(T[i]-Tsol)/(Tliq-Tsol);
            scalar S=-Cu*sqr(1.0-fl)/(pow3(fl)+eps);
            eqn.diag()[i]+=S*rhoRef*V[i];
        }
//}}} end code
}


void
Foam::fv::
mushyZoneDarcyFvOptionvectorSource::constrain
(
    fvMatrix<vector>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "mushyZoneDarcyFvOptionvectorSource::constrain()\n";
    }

//{{{ begin code
    
//}}} end code
}


// ************************************************************************* //

