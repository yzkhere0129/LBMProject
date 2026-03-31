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
#line 82 "/home/yzk/OpenFOAM/spot_melting_marangoni/constant/fvOptions/phaseChangeSource"
#include "fvcDdt.H"
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
// SHA1 = e841af48af259d06810ffa213b539ac525b5cfb7
//
// unique function name that can be checked if the correct library version
// has been loaded
extern "C" void mushyZonePhaseChange_e841af48af259d06810ffa213b539ac525b5cfb7(bool load)
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

defineTypeNameAndDebug(mushyZonePhaseChangeFvOptionscalarSource, 0);
addRemovableToRunTimeSelectionTable
(
    option,
    mushyZonePhaseChangeFvOptionscalarSource,
    dictionary
);

} // End namespace fv
} // End namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::fv::
mushyZonePhaseChangeFvOptionscalarSource::
mushyZonePhaseChangeFvOptionscalarSource
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
        printMessage("Construct mushyZonePhaseChange fvOption from dictionary");
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::fv::
mushyZonePhaseChangeFvOptionscalarSource::
~mushyZonePhaseChangeFvOptionscalarSource()
{
    if (false)
    {
        printMessage("Destroy mushyZonePhaseChange");
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void
Foam::fv::
mushyZonePhaseChangeFvOptionscalarSource::correct
(
    GeometricField<scalar, fvPatchField, volMesh>& fld
)
{
    if (false)
    {
        Info<< "mushyZonePhaseChangeFvOptionscalarSource::correct()\n";
    }

//{{{ begin code
    
//}}} end code
}


void
Foam::fv::
mushyZonePhaseChangeFvOptionscalarSource::addSup
(
    fvMatrix<scalar>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "mushyZonePhaseChangeFvOptionscalarSource::addSup()\n";
    }

//{{{ begin code - warn/fatal if not implemented?
    #line 89 "/home/yzk/OpenFOAM/spot_melting_marangoni/constant/fvOptions/phaseChangeSource"
const scalar Tsol=1650, Tliq=1700, Lf=260000, rhoVal=7900;
        const scalar dt=mesh().time().deltaTValue();
        const volScalarField& T=mesh().lookupObject<volScalarField>("T");

        const word lfN("liquidFraction");
        volScalarField* p=mesh().getObjectPtr<volScalarField>(lfN);
        if(!p){p=new volScalarField(IOobject(lfN,mesh().time().timeName(),mesh(),
            IOobject::NO_READ,IOobject::AUTO_WRITE,IOobject::REGISTER),
            mesh(),dimensionedScalar("lf",dimless,0));p->store();}
        volScalarField& lf=*p; lf.oldTime();

        static label li=-1;
        if(mesh().time().timeIndex()!=li){
            forAll(lf,i){
                if(T[i]<=Tsol) lf[i]=0;
                else if(T[i]>=Tliq) lf[i]=1;
                else lf[i]=(T[i]-Tsol)/(Tliq-Tsol);
            } li=mesh().time().timeIndex();
        }
        const scalarField& V=mesh().V();
        const scalarField& lo=lf.oldTime().primitiveField();
        forAll(eqn.source(),i){
            eqn.source()[i]+=rhoVal*Lf*(lf[i]-lo[i])/dt*V[i];
        }
//}}} end code
}


void
Foam::fv::
mushyZonePhaseChangeFvOptionscalarSource::addSup
(
    const volScalarField& rho,
    fvMatrix<scalar>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "mushyZonePhaseChangeFvOptionscalarSource::addSup(rho)\n";
    }

//{{{ begin code - warn/fatal if not implemented?
    #line 117 "/home/yzk/OpenFOAM/spot_melting_marangoni/constant/fvOptions/phaseChangeSource"
const scalar Tsol=1650, Tliq=1700, Lf=260000, rhoVal=7900;
        const scalar dt=mesh().time().deltaTValue();
        const volScalarField& T=mesh().lookupObject<volScalarField>("T");

        const word lfN("liquidFraction");
        volScalarField* p=mesh().getObjectPtr<volScalarField>(lfN);
        if(!p){p=new volScalarField(IOobject(lfN,mesh().time().timeName(),mesh(),
            IOobject::NO_READ,IOobject::AUTO_WRITE,IOobject::REGISTER),
            mesh(),dimensionedScalar("lf",dimless,0));p->store();}
        volScalarField& lf=*p; lf.oldTime();

        static label li=-1;
        if(mesh().time().timeIndex()!=li){
            forAll(lf,i){
                if(T[i]<=Tsol) lf[i]=0;
                else if(T[i]>=Tliq) lf[i]=1;
                else lf[i]=(T[i]-Tsol)/(Tliq-Tsol);
            } li=mesh().time().timeIndex();
        }
        const scalarField& V=mesh().V();
        const scalarField& lo=lf.oldTime().primitiveField();
        forAll(eqn.source(),i){
            eqn.source()[i]+=rhoVal*Lf*(lf[i]-lo[i])/dt*V[i];
        }
//}}} end code
}


void
Foam::fv::
mushyZonePhaseChangeFvOptionscalarSource::constrain
(
    fvMatrix<scalar>& eqn,
    const label fieldi
)
{
    if (false)
    {
        Info<< "mushyZonePhaseChangeFvOptionscalarSource::constrain()\n";
    }

//{{{ begin code
    
//}}} end code
}


// ************************************************************************* //

