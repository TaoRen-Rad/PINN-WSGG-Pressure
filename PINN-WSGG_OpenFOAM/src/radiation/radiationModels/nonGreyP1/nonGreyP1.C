/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2020 OpenCFD Ltd.
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

#include "nonGreyP1.H"
#include "fvmLaplacian.H"
#include "fvmSup.H"
#include "absorptionEmissionModel.H"
#include "extrapolatedCalculatedFvPatchFields.H"
#include "scatterModel.H"
#include "constants.H"
#include "addToRunTimeSelectionTable.H"
#include "globalModel.H"
#include "clockTime.H"          // 新增：用于计时
#include "PstreamReduceOps.H"
#include "OFstream.H"

using namespace Foam::constant;

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    namespace radiation
    {
        defineTypeNameAndDebug(nonGreyP1, 0);
        addToRadiationRunTimeSelectionTables(nonGreyP1);
    }
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::radiation::nonGreyP1::nonGreyP1(const volScalarField& T)
:
    radiationModel(typeName, T),
    coeffsDict_(coeffs_),
    G_
    (
        IOobject("G", mesh_.time().timeName(), mesh_, IOobject::MUST_READ, IOobject::AUTO_WRITE),
        mesh_
    ),
    qr_
    (
        IOobject("qr", mesh_.time().timeName(), mesh_, IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE),
        mesh_,
        dimensionedScalar(dimMass/pow3(dimTime), Zero)
    ),
    a_
    (
        IOobject("a", mesh_.time().timeName(), mesh_, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh_,
        dimensionedScalar(dimless/dimLength, Zero)
    ),
    e_
    (
        IOobject("e", mesh_.time().timeName(), mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh_,
        dimensionedScalar(dimless/dimLength, Zero)
    ),
    E_
    (
        IOobject("E", mesh_.time().timeName(), mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh_,
        dimensionedScalar(dimMass/dimLength/pow3(dimTime), Zero)
    ),
    a0_(coeffsDict_.getOrDefault<scalar>("a0", 1e-6)),
    Gg_(absorptionEmission_->nBands())
{
    initGg();
}


Foam::radiation::nonGreyP1::nonGreyP1(const dictionary& dict, const volScalarField& T)
:
    radiationModel(typeName, dict, T),
    coeffsDict_(dict.optionalSubDict(typeName + "Coeffs")),
    G_
    (
        IOobject("G", mesh_.time().timeName(), mesh_, IOobject::MUST_READ, IOobject::AUTO_WRITE),
        mesh_
    ),
    qr_
    (
        IOobject("qr", mesh_.time().timeName(), mesh_, IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE),
        mesh_,
        dimensionedScalar(dimMass/pow3(dimTime), Zero)
    ),
    a_
    (
        IOobject("a", mesh_.time().timeName(), mesh_, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh_,
        dimensionedScalar(dimless/dimLength, Zero)
    ),
    e_
    (
        IOobject("e", mesh_.time().timeName(), mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh_,
        dimensionedScalar(dimless/dimLength, Zero)
    ),
    E_
    (
        IOobject("E", mesh_.time().timeName(), mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh_,
        dimensionedScalar(dimMass/dimLength/pow3(dimTime), Zero)
    ),
    a0_(coeffsDict_.getOrDefault<scalar>("a0", 1e-6)),
    Gg_(absorptionEmission_->nBands())
{
    initGg();
}

// * * * * * * * * * * * * * Private member Functions  * * * * * * * * * * * * //

void Foam::radiation::nonGreyP1::initGg()
{
    forAll(Gg_, bandI)
    {
        string GgFileInit("Gg.");
        OStringStream Convert;
        Convert << bandI;
        Gg_.set
        (
            bandI, 
            new volScalarField 
            (
                IOobject
                (
                    GgFileInit + Convert.str(), 
                    G_.time().timeName(),
                    G_.mesh(), 
                    IOobject::READ_IF_PRESENT,
                    IOobject::AUTO_WRITE
                ),
                G_
            )
        );
    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::radiation::nonGreyP1::read()
{
    if (!radiationModel::read())
    {
        return false;
    }

    coeffsDict_ = coeffs_.optionalSubDict(typeName + "Coeffs");
    a0_ = coeffsDict_.getOrDefault<scalar>("a0", 1e-6);

    return true;
}


void Foam::radiation::nonGreyP1::calculate()
{
    // ====================== 时间步控制（防止重复求解） ======================
    if (solverFreq_ > 0 && (mesh_.time().timeIndex() % solverFreq_ != 0))
    {
        return;
    }

    static label lastSolvedTimeIndex = -1;
    if (lastSolvedTimeIndex == mesh_.time().timeIndex())
    {
        return;
    }
    lastSolvedTimeIndex = mesh_.time().timeIndex();

    // ====================== 时间统计 ======================
    clockTime totalTimer;                    // 整个 calculate() 时间
    clockTime spectralTimer;                 // Spectral (WSGG) 时间
    clockTime rteTimer;                      // RTE 求解时间

    scalar spectralTime = 0.0;
    scalar rteTime = 0.0;

    // ====================== 残差日志 ======================
    const fileName postDir(mesh_.time().path()/"postProcessing");
    const fileName residualDir(postDir/"radiationResiduals");
    mkDir(postDir);
    mkDir(residualDir);
    OFstream residualLog(residualDir/("GgResiduals_" + mesh_.time().timeName() + ".dat"));
    residualLog
        << "# band initialResidual finalResidual iterations status" << nl;
    // 1. Spectral 部分（WSGG + absorptionEmission）
    spectralTimer.timeIncrement();
    absorptionEmission_->correct(G_, Gg_);
    spectralTime = spectralTimer.timeIncrement();

    // 2. RTE 求解部分（每个 band 的 P1 方程）
    rteTimer.timeIncrement();
    globalModel* spectralModel = globalModel::getInstance();
    double* wNq = spectralModel->getwNq();

    forAll(Gg_, bandI)
    {
        spectralModel->setBandI(bandI);
        a_ = absorptionEmission_->a(bandI);
        e_ = absorptionEmission_->e(bandI);
        E_ = absorptionEmission_->E(bandI);
        const volScalarField sigmaEff(scatter_->sigmaEff());

        const dimensionedScalar a0("a0", a_.dimensions(), a0_);

        const volScalarField gamma
        (
            IOobject
            (
                "gammaRad",
                G_.mesh().time().timeName(),
                G_.mesh(),
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            1.0/(3.0*a_ + sigmaEff + a0)
        );

        const scalar aMax = gMax(a_);
        const scalar EMax = gMax(E_);

        if (aMax <= SMALL && EMax <= SMALL)
        {
            Gg_[bandI] = dimensionedScalar(Gg_[bandI].dimensions(), Zero);
            residualLog << bandI << " 0 0 0 skipped" << nl;
            Info << "Gg." << bandI << ": skipped clear-gas band" << endl;
            continue;
        }

        const solverPerformance perf
        (
            solve
            (
                fvm::laplacian(gamma, Gg_[bandI])
              - fvm::Sp(a_, Gg_[bandI])
             ==
               -4.0*(e_*a_*physicoChemical::sigma*pow4(T_)) * dimensionedScalar(dimLength, 1) - E_
            )
        );

        residualLog
            << bandI << ' '
            << perf.initialResidual() << ' '
            << perf.finalResidual() << ' '
            << perf.nIterations() << ' '
            << "solved" << nl;
        Info << "Gg." << bandI
             << ": Initial residual = " << perf.initialResidual()
             << ", Final residual = " << perf.finalResidual()
             << ", No Iterations " << perf.nIterations() << endl;
    }
    rteTime = rteTimer.timeIncrement();

    // ====================== 输出时间统计 ======================
    scalar totalTime = totalTimer.timeIncrement();

    Info << "=== Radiation Timing Statistics ===" << nl
         << "  Spectral (WSGG + absorption) : " << spectralTime << " s" << nl
         << "  RTE solve (P1 equations)     : " << rteTime << " s" << nl
         << "  Total in calculate()         : " << totalTime << " s" << nl
         << "  RTE / Total ratio            : " << (rteTime / totalTime * 100) << " %" << endl;

    // ====================== 后续处理 ======================
    G_ = 0.0*G_;

    forAll(Gg_, bandI)
    {
        G_ += Gg_[bandI]*wNq[bandI];
        forAll(mesh_.boundaryMesh(), patchi)
        {
            G_.boundaryFieldRef()[patchi] += Gg_[bandI].boundaryField()[patchi]*wNq[bandI];
        }
    }

    spectralModel->updateRu(Ru());
    spectralModel->updateRp(Rp());
}

Foam::tmp<Foam::volScalarField> Foam::radiation::nonGreyP1::Rp() const
{
    tmp<volScalarField> tRp
    (
        new volScalarField
        (
            IOobject("Rp", mesh_.time().timeName(), mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
            4.0*absorptionEmission_->eCont()*physicoChemical::sigma*0.0
        )
    );

    globalModel* spectralModel = globalModel::getInstance();

    forAll(Gg_, bandI)
    {
        tRp.ref() += spectralModel->getwNq()[bandI] 
                   * absorptionEmission_->a(bandI) 
                   * 4*physicoChemical::sigma 
                   * absorptionEmission_->e(bandI) 
                   * dimensionedScalar(dimLength, 1);
    }

    return tRp;
}


Foam::tmp<Foam::DimensionedField<Foam::scalar, Foam::volMesh>>
Foam::radiation::nonGreyP1::Ru() const
{
    tmp<DimensionedField<scalar, volMesh>> tRu
    (
        new volScalarField
        (
            IOobject("Ru", mesh_.time().timeName(), mesh_, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh_,
            dimensionedScalar(dimMass/dimLength/pow3(dimTime), Zero)
        )
    );

    globalModel* spectralModel = globalModel::getInstance();

    forAll(Gg_, bandI)
    {
        tRu.ref() += spectralModel->getwNq()[bandI] 
                   * absorptionEmission_->a(bandI) 
                   * Gg_[bandI];
    }

    return tRu;
}


Foam::label Foam::radiation::nonGreyP1::nBands() const
{
    return absorptionEmission_->nBands();
}

// ************************************************************************* //
