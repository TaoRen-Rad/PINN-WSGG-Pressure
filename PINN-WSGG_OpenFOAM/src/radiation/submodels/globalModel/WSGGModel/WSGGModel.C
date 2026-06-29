#include "WSGGModel.H"
#include "basicSpecieMixture.H"

namespace
{
Foam::scalar moleFractionOrZero
(
    const Foam::basicSpecieMixture& mixture,
    const Foam::label celli,
    const Foam::scalar invWt,
    const Foam::word& specieName
)
{
    if (!mixture.species().contains(specieName))
    {
        return 0.0;
    }

    const Foam::label specieI = mixture.species()[specieName];
    return mixture.Y(specieI)[celli]/(mixture.W(specieI)*invWt);
}


Foam::scalar moleFractionOrZero
(
    const Foam::basicSpecieMixture& mixture,
    const Foam::label patchi,
    const Foam::label facei,
    const Foam::scalar invWt,
    const Foam::word& specieName
)
{
    if (!mixture.species().contains(specieName))
    {
        return 0.0;
    }

    const Foam::label specieI = mixture.species()[specieName];
    return
        mixture.Y(specieI).boundaryField()[patchi][facei]
       /(mixture.W(specieI)*invWt);
}
}

Foam::radiation::WSGGModel::WSGGModel(word spectrumModelName, label nBands, const fvMesh& mesh)
:
    globalModel(spectrumModelName, nBands, mesh),
    wsggMLPParameter_()
{
    Info << "Selecting " << spectrumModelName_ << " as the spectral model" << endl;
    if (nBands_ != 9)
    {
        WarningInFunction
            << "Current deployed WSGG MLP returns 9 grey-gas weights plus 9 "
            << "absorption coefficients. Running with nBands=" << nBands_
            << " will slice that 9-band output and is not the native layout."
            << endl;
    }
    initKA();
    Tref_ = 1000.0;

    gNqDB = new double[32];
    gNq   = new double[32];
    wNqDB = new double[32];
    wNq   = new double[32];
    for(int i=0; i < nBands_; i++){
        gNqDB[i] = double(i) / double(nBands_-1);
        gNq[i]   = double(i) / double(nBands_-1);
        wNqDB[i] = 1.0;
        wNq[i]   = 1.0;
    }

    Info << "Initial kdist Information (will be updated during simulation)" << endl;
    Info << "gNq :";
    for(int i = 0; i < nBands; i++)
    {
        Info << gNq[i] << "\t";
    }
    Info << endl << "wNq :";
    for(int i = 0; i < nBands; i++)
    {
        Info << wNq[i] << "\t";
    }
    Info << endl;
}

void Foam::radiation::WSGGModel::updateTref()
{
    // Dummy implementation - WSGG do not have Tref
}

void Foam::radiation::WSGGModel::updateKA()
{
    const basicSpecieMixture& mixture =
        dynamic_cast<const basicSpecieMixture&>(thermo_);

    const volScalarField& T = thermo_.T();
    const volScalarField& p = thermo_.p();
    const bool pressureAwareModel = (wsggMLPParameter_.input_size() == 5);

    // 用于统计平均权重
    scalarField avgWeights(nBands_, 0.0);
    scalar totalCells = 0.0;
    scalar minSpeciesSum = GREAT;
    scalar maxSpeciesSum = -GREAT;
    scalar minTemp = GREAT;
    scalar maxTemp = -GREAT;
    scalar minPressureBar = GREAT;
    scalar maxPressureBar = -GREAT;

    forAll(T, celli)
    {
        scalar invWt = 0.0;
        scalar xco2_cell, xh2o_cell, xco_cell;  // NH3 改为 CO
        forAll(mixture.Y(), s)
        {
            invWt += mixture.Y(s)[celli]/mixture.W(s);
        }
        xco2_cell = moleFractionOrZero(mixture, celli, invWt, "CO2");
        xh2o_cell = moleFractionOrZero(mixture, celli, invWt, "H2O");
        xco_cell = moleFractionOrZero(mixture, celli, invWt, "CO");
        const scalar xsum = xh2o_cell + xco2_cell + xco_cell;

        minSpeciesSum = min(minSpeciesSum, xsum);
        maxSpeciesSum = max(maxSpeciesSum, xsum);
        minTemp = min(minTemp, T[celli]);
        maxTemp = max(maxTemp, T[celli]);
        minPressureBar = min(minPressureBar, p[celli]/1e5);
        maxPressureBar = max(maxPressureBar, p[celli]/1e5);

        std::vector<double> input_wsgg;
        if (pressureAwareModel)
        {
            input_wsgg = {T[celli], xh2o_cell, xco2_cell, xco_cell, p[celli]/1e5};
        }
        else
        {
            input_wsgg = {T[celli], xh2o_cell, xco2_cell, xco_cell};
        }
        auto output = wsggMLPParameter_.get_prediction(input_wsgg);
        forAll(ki_, bandI)
        {
            ai_[bandI][celli] = output[bandI];
            ki_[bandI][celli] = output[nBands_ + bandI] * xsum * (p[celli]/1e5) * 100.0;
            
            // 累加权重用于计算平均值
            avgWeights[bandI] += output[bandI];
        }
        totalCells += 1.0;
    }
    
    forAll(T.boundaryField(), patchi)
    {
        forAll (T.boundaryField()[patchi], facei)
        {
            scalar invWt = 0.0;
            scalar xco2_cell, xh2o_cell, xco_cell;  // NH3 改为 CO
            forAll(mixture.Y(), s)
            {
                invWt += mixture.Y(s).boundaryField()[patchi][facei]/mixture.W(s);
            }
            xco2_cell = moleFractionOrZero(mixture, patchi, facei, invWt, "CO2");
            xh2o_cell = moleFractionOrZero(mixture, patchi, facei, invWt, "H2O");
            xco_cell = moleFractionOrZero(mixture, patchi, facei, invWt, "CO");
            const scalar xsum = xh2o_cell + xco2_cell + xco_cell;

            std::vector<double> input_wsgg;
            if (pressureAwareModel)
            {
                input_wsgg =
                    {
                        T.boundaryField()[patchi][facei],
                        xh2o_cell,
                        xco2_cell,
                        xco_cell,
                        p.boundaryField()[patchi][facei]/1e5
                    };
            }
            else
            {
                input_wsgg =
                    {
                        T.boundaryField()[patchi][facei],
                        xh2o_cell,
                        xco2_cell,
                        xco_cell
                    };
            }
            auto output = wsggMLPParameter_.get_prediction(input_wsgg);
            forAll(ai_, bandI)
            {
                ai_[bandI].boundaryFieldRef()[patchi][facei] = output[bandI];
                ki_[bandI].boundaryFieldRef()[patchi][facei] =
                    output[nBands_ + bandI]
                  * xsum
                  * (p.boundaryField()[patchi][facei]/1e5)
                  * 100.0;
            }
        }
    }

    // 计算并行归约,获取全局平均值
    if (Pstream::parRun())
    {
        reduce(avgWeights, sumOp<scalarField>());
        reduce(totalCells, sumOp<scalar>());
        reduce(minSpeciesSum, minOp<scalar>());
        reduce(maxSpeciesSum, maxOp<scalar>());
        reduce(minTemp, minOp<scalar>());
        reduce(maxTemp, maxOp<scalar>());
        reduce(minPressureBar, minOp<scalar>());
        reduce(maxPressureBar, maxOp<scalar>());
    }

    // 计算平均权重并显示
    if (totalCells > 0)
    {
        avgWeights /= totalCells;
        
        Info << "WSGG input envelope:" << endl;
        Info << "  T [K] range              : " << minTemp << " .. " << maxTemp << endl;
        Info << "  p [bar] range            : " << minPressureBar << " .. " << maxPressureBar << endl;
        Info << "  (xH2O+xCO2+xCO) range    : " << minSpeciesSum << " .. " << maxSpeciesSum << endl;
        Info << "WSGG Gray Gas Weights (domain average):" << endl;
        scalar sumWeights = 0.0;
        for(label i = 0; i < nBands_; i++)
        {
            Info << "  Band " << i << ": a" << i << " = " << avgWeights[i];
            sumWeights += avgWeights[i];
            if (i < nBands_ - 1)
            {
                Info << endl;
            }
        }
        Info << endl;
        Info << "  Sum of weights: " << sumWeights << endl;
        Info << "WSGG band coefficient envelope:" << endl;
        for (label i = 0; i < nBands_; i++)
        {
            const scalar aiMin = gMin(ai_[i]);
            const scalar aiMax = gMax(ai_[i]);
            const scalar kiMin = gMin(ki_[i]);
            const scalar kiMax = gMax(ki_[i]);
            Info << "  Band " << i
                 << " : a[min,max]=(" << aiMin << ", " << aiMax << ")"
                 << ", k[min,max]=(" << kiMin << ", " << kiMax << ")"
                 << endl;
        }
        Info << endl;
    }
}

Foam::tmp<Foam::volScalarField> Foam::radiation::WSGGModel::get_k(const label bandI)
{
    ki_[bandI].correctBoundaryConditions();
    return ki_[bandI];
}

Foam::tmp<Foam::volScalarField> Foam::radiation::WSGGModel::get_a(const label bandI)
{
    ai_[bandI].correctBoundaryConditions();
    return ai_[bandI];
}
// 氨
// #include "WSGGModel.H"
// #include "basicSpecieMixture.H"

// Foam::radiation::WSGGModel::WSGGModel(word spectrumModelName, label nBands, const fvMesh& mesh)
// :
//     globalModel(spectrumModelName, nBands, mesh),
//     wsggMLPParameter_()
// {
//     Info << "Selecting " << spectrumModelName_ << " as the spectral model" << endl;
//     initKA();
//     Tref_ = 1000.0;

//     gNqDB = new double[32];
//     gNq = new double[32];
//     wNqDB = new double[32];
//     wNq = new double[32];
    
//     for(int i=0; i < nBands_; i++){
//         gNqDB[i] = double(i) / double(nBands_-1);
//         gNq[i] = double(i) / double(nBands_-1);
//         wNqDB[i] = 1.0;
//         wNq[i] = 1.0;
//     }

//     Info << "Dummy kdist Information" << endl;
//     Info << "gNq :";
//     for(int i = 0; i < nBands; i++)
//     {
//         Info << gNq[i] << "\t";
//     }
//     Info << endl << "wNq :";
//     for(int i = 0; i < nBands; i++)
//     {
//         Info << wNq[i] << "\t";
//     }
// }

// void Foam::radiation::WSGGModel::updateTref()
// {
//     // Dummy implementation - WSGG do not have Tref
// }

// void Foam::radiation::WSGGModel::updateKA()
// {
//     const basicSpecieMixture& mixture =
//     dynamic_cast<const basicSpecieMixture&>(thermo_);

//     const volScalarField& T = thermo_.T();
//     // const volScalarField& p = thermo_.p();

//     forAll(T, celli)
//     {
//         scalar invWt = 0.0;
//         scalar xco2_cell, xh2o_cell, xnh3_cell;
//         forAll(mixture.Y(), s)
//         {
//             invWt += mixture.Y(s)[celli]/mixture.W(s);
//         }
//         xco2_cell = mixture.Y("CO2")[celli]/(mixture.W(mixture.species()["CO2"])*invWt);
//         xh2o_cell = mixture.Y("H2O")[celli]/(mixture.W(mixture.species()["H2O"])*invWt);
//         xnh3_cell = mixture.Y("NH3")[celli]/(mixture.W(mixture.species()["NH3"])*invWt);

//         std::vector<double> input_wsgg = {T[celli], xh2o_cell, xco2_cell, xnh3_cell};
//         auto output = wsggMLPParameter_.get_prediction(input_wsgg);
//         forAll(ki_, bandI)
//         {
//             ai_[bandI][celli] = output[bandI];
//             ki_[bandI][celli] = output[nBands_ + bandI] * (xh2o_cell + xco2_cell + xnh3_cell) * 100.0;
//         }
//     }
//     forAll(T.boundaryField(), patchi)
//     {
//         forAll (T.boundaryField()[patchi], facei)
//         {
//             scalar invWt = 0.0;
//             scalar xco2_cell, xh2o_cell, xnh3_cell;
//             forAll(mixture.Y(), s)
//             {
//                 invWt += mixture.Y(s).boundaryField()[patchi][facei]/mixture.W(s);
//             }
//             xco2_cell = mixture.Y("CO2").boundaryField()[patchi][facei]/(mixture.W(mixture.species()["CO2"])*invWt);
//             xh2o_cell = mixture.Y("H2O").boundaryField()[patchi][facei]/(mixture.W(mixture.species()["H2O"])*invWt);
//             xnh3_cell = mixture.Y("NH3").boundaryField()[patchi][facei]/(mixture.W(mixture.species()["NH3"])*invWt);

//             std::vector<double> input_wsgg = {T.boundaryField()[patchi][facei], xh2o_cell, xco2_cell, xnh3_cell};
//             auto output = wsggMLPParameter_.get_prediction(input_wsgg);
//             forAll(ai_, bandI)
//             {
//                 ai_[bandI].boundaryFieldRef()[patchi][facei] = output[bandI];
//                 ki_[bandI].boundaryFieldRef()[patchi][facei] = output[nBands_ + bandI] * (xh2o_cell + xco2_cell + xnh3_cell) * 100.0;
//             }
//         }
//     }
// }

// Foam::tmp<Foam::volScalarField> Foam::radiation::WSGGModel::get_k(const label bandI)
// {
//     ki_[bandI].correctBoundaryConditions();
//     return ki_[bandI];
// }

// Foam::tmp<Foam::volScalarField> Foam::radiation::WSGGModel::get_a(const label bandI)
// {
//     ai_[bandI].correctBoundaryConditions();
//     return ai_[bandI];
// }
