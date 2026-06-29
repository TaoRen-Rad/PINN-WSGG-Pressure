#ifndef WSGGMLP_H
#define WSGGMLP_H

#include "mlp.h"
#include <vector>
#include <string>


class wsggmlpWSGG : public MLP
{
public:
    wsggmlpWSGG(const std::string& json_file);

    std::vector<double> predict(const std::vector<double>& input) override;
};


#endif
//双mlp
// #ifndef WSGGMLP_H
// #define WSGGMLP_H

// #include "mlp.h"
// #include <vector>
// #include <memory>

// class wsggmlpWeights : public MLP {
// public:
//     wsggmlpWeights(const std::string& json_file);
//     std::vector<double> predict(const std::vector<double>& input) override;
// };

// class wsggmlpAbscs : public MLP {
// public:
//     wsggmlpAbscs(const std::string& json_file);
//     std::vector<double> predict(const std::vector<double>& input) override;
// };

// #endif
