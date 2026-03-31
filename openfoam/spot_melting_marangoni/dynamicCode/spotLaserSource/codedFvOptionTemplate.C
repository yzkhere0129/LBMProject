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
// SHA1 = ce928d5cb3180c087de9bebe70200bb92a039ab5
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void spotLaserSource_ce928d5cb3180c087de9bebe70200bb92a039ab5(bool load)
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

defineTypeNameAndDebug(spotLaserSourceFvOptionscalarSource, 0);
addRemovableToRunTimeSelectionTable
(
    option,
    spotLaserSourceFvOptionscalarSource,
    dictionary
);

} // End namespace fv
} // End namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::fv::
spotLaserSourceFvOptionscalarSource::
spotLaserSourceFvOptionscalarSource
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
        printMessage("Construct spotLaserSource fvOption from dictionary");
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::fv::
spotLaserSourceFvOptionscalarSource::
~spotLaserSourceFvOptionscalarSource()
{
    if (false)
    {
        printMessage("Destroy spotLaserSource");
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void
Foam::fv::
spotLaserSourceFvOptionscalarSource::correct
(
    GeometricField<scalar, fvPatchField, volMesh>& fld
)
{
    if (false)
    {
        Info<< "spotLaserSourceFvOptionscalarSource::correct()\n";
    }

//{{{ begin code
    
//}}} end code
}


void
Foam::fv::
spotLaserSourceFvOptionscalarSource::addSup
(
    fvMatrix<scalar>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "spotLaserSourceFvOptionscalarSource::addSup()\n";
    }

//{{{ begin code - warn/fatal if not implemented?
    #line 33 "/home/yzk/OpenFOAM/spot_melting_marangoni/constant/fvOptions/laserSource"
const scalar P=150, r0=25e-6, eta=0.35, t_on=50e-6, dy_s=2e-6;
        const scalar t=mesh().time().value();
        if(t>=t_on) return;
        const volVectorField& C=mesh().C();
        const scalar pi=Foam::constant::mathematical::pi;
        const scalar Q0=2.0*eta*P/(pi*sqr(r0));
        forAll(eqn.source(),i)
        {
            if(C[i].y()>-dy_s)
            {
                scalar r2=sqr(C[i].x())+sqr(C[i].z());
                eqn.source()[i]-=Q0*Foam::exp(-2.0*r2/sqr(r0))/dy_s*mesh().V()[i];
            }
        }
//}}} end code
}


void
Foam::fv::
spotLaserSourceFvOptionscalarSource::addSup
(
    const volScalarField& rho,
    fvMatrix<scalar>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "spotLaserSourceFvOptionscalarSource::addSup(rho)\n";
    }

//{{{ begin code - warn/fatal if not implemented?
    #line 51 "/home/yzk/OpenFOAM/spot_melting_marangoni/constant/fvOptions/laserSource"
const scalar P=150, r0=25e-6, eta=0.35, t_on=50e-6, dy_s=2e-6;
        const scalar t=mesh().time().value();
        if(t>=t_on) return;
        const volVectorField& C=mesh().C();
        const scalar pi=Foam::constant::mathematical::pi;
        const scalar Q0=2.0*eta*P/(pi*sqr(r0));
        forAll(eqn.source(),i)
        {
            if(C[i].y()>-dy_s)
            {
                scalar r2=sqr(C[i].x())+sqr(C[i].z());
                eqn.source()[i]-=Q0*Foam::exp(-2.0*r2/sqr(r0))/dy_s*mesh().V()[i];
            }
        }
//}}} end code
}


void
Foam::fv::
spotLaserSourceFvOptionscalarSource::constrain
(
    fvMatrix<scalar>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "spotLaserSourceFvOptionscalarSource::constrain()\n";
    }

//{{{ begin code
    
//}}} end code
}


// ************************************************************************* //

