#include "globalModel.H"
#include "WSGGModel.H"

Foam::radiation::globalModel* Foam::radiation::globalModel::instance = nullptr;

Foam::radiation::globalModel::globalModel(word spectrumModelName, label nBands, const fvMesh& mesh)
:
    spectrumModelName_(spectrumModelName),
    nBands_(nBands),
    mesh_(mesh),
    thermo_(mesh.lookupObject<fluidThermo>(basicThermo::dictName)),
    writeRu_
    (
        IOobject
        (
            "debug_Ru",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimMass/dimLength/pow3(dimTime), Zero)
    ),
    writeRp_
    (
        IOobject
        (
            "debug_Rp",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimMass/dimLength/pow3(dimTime)/pow4(dimTemperature), Zero)
    ),
    bandI_(0)
{
    // Quadrature initialization moved to derived classes
}

void Foam::radiation::globalModel::initKA()
{
    ki_.setSize(nBands_);
    ai_.setSize(nBands_);
    for (label i=0; i<nBands_; i++)
    {
        ki_.set(i, new volScalarField
        (
            IOobject
            (
                "k" + std::to_string(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh_,
            dimensionedScalar(dimless/dimLength, ROOTVSMALL)
        ));
        ai_.set(i, new volScalarField
        (
            IOobject
            (
                "a" + std::to_string(i),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh_,
            dimensionedScalar(dimless/dimLength, ROOTVSMALL)
        ));
    }
}

Foam::radiation::globalModel* Foam::radiation::globalModel::getInstance()
{
    if (!instance)
    {
        FatalErrorInFunction
            << "Global model instance not yet created"
            << abort(FatalError);
    }
    return instance;
}

Foam::radiation::globalModel* Foam::radiation::globalModel::getInstance(word spectrumModelName, label nBands, const fvMesh& mesh)
{
    if (!instance)
    {
        if (spectrumModelName == "WSGG")
        {
            instance = new WSGGModel(spectrumModelName, nBands, mesh);
        }
        else
        {
            FatalErrorInFunction
                << "Unsupported spectrum model in this WSGG/P1 distribution: "
                << spectrumModelName << nl
                << "Only \"WSGG\" is available."
                << abort(FatalError);
        }
    }
    return instance;
}

Foam::word Foam::radiation::globalModel::getSpectrumModelName()
{
    return spectrumModelName_;
}

Foam::label Foam::radiation::globalModel::getNBands()
{
    return nBands_;
}

Foam::scalar Foam::radiation::globalModel::getTref()
{
    return Tref_;
}

double* Foam::radiation::globalModel::getwNq()
{
    return wNq;
}

void Foam::radiation::globalModel::updateRu(const DimensionedField<scalar, volMesh>& Ru)
{
    writeRu_ = Ru;
}

void Foam::radiation::globalModel::updateRp(const volScalarField Rp)
{
    writeRp_ = Rp;
}

void Foam::radiation::globalModel::writeDebugFields()
{
    writeRu_.write();
    writeRp_.write();
}

void Foam::radiation::globalModel::setBandI(const label bandI)
{
    bandI_ = bandI;
}

Foam::label Foam::radiation::globalModel::getBandI()
{
    return bandI_;
}
