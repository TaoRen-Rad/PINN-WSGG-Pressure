/*---------------------------------------------------------------------------*\
Abstract base class for global radiation models using singleton pattern.
\*---------------------------------------------------------------------------*/

#ifndef globalModel_H
#define globalModel_H

#include "volFields.H"
#include "fluidThermo.H"

namespace Foam
{
namespace radiation
{

class globalModel
{
protected:
    // Protected data members shared by all models
    word spectrumModelName_;
    label nBands_;
    scalar Tref_;
    scalar xco2_ref_;
    scalar xh2o_ref_;
    scalar xco_ref_;
    const fvMesh& mesh_;
    const fluidThermo& thermo_;
    PtrList<volScalarField> ki_;
    PtrList<volScalarField> ai_;
    DimensionedField<scalar, volMesh> writeRu_;
    volScalarField writeRp_;
    label bandI_;

    // Quadrature data
    double* gNqDB;
    double* wNqDB;
    double* gNq;
    double* wNq;

    // Singleton instance
    static globalModel* instance;  // Still stored as pointer but accessed via reference

    // Protected constructor for derived classes
    globalModel(word spectrumModelName, label nBands, const fvMesh& mesh);

    // Initialization helper
    void initKA();

public:
    virtual ~globalModel() = default;

    // Singleton access
    static globalModel* getInstance();
    static globalModel* getInstance(word spectrumModelName, label nBands, const fvMesh& mesh);

    // Pure virtual interface methods
    virtual void updateTref() = 0;
    virtual void updateKA() = 0;
    virtual tmp<volScalarField> get_k(const label bandI) = 0;
    virtual tmp<volScalarField> get_a(const label bandI) = 0;

    // Common getters
    word getSpectrumModelName();
    label getNBands();
    scalar getTref();
    double* getwNq();
    void updateRu(const DimensionedField<scalar, volMesh>& Ru);
    void updateRp(const volScalarField Rp);
    void writeDebugFields();
    void setBandI(const label bandI);
    label getBandI();
};

} // namespace radiation
} // namespace Foam

#endif
