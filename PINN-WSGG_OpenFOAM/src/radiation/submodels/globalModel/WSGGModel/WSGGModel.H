/*---------------------------------------------------------------------------*\
WSGG implementation of the globalModel interface (dummy placeholder)
\*---------------------------------------------------------------------------*/

#ifndef WSGGModel_H
#define WSGGModel_H

#include "globalModel.H"
#include "wsggMLPParameter.h"

namespace Foam
{
namespace radiation
{

class WSGGModel : public globalModel
{
private:
    wsggMLPParameter wsggMLPParameter_;
public:
    WSGGModel(word spectrumModelName, label nBands, const fvMesh& mesh);

    // Implement pure virtual methods (dummy implementations)
    virtual void updateTref() override;
    virtual void updateKA() override;
    virtual tmp<volScalarField> get_k(const label bandI) override;
    virtual tmp<volScalarField> get_a(const label bandI) override;
};

} // namespace radiation
} // namespace Foam

#endif
