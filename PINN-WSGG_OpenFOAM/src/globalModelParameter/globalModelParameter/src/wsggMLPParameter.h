#ifndef WSGGMLPPARAMETER_H
#define WSGGMLPPARAMETER_H

#include "wsggmlp.h"

#include <vector>
#include <string>
#include <memory>

class wsggMLPParameter
{
private:
    std::vector<double> x_mean;
    std::vector<double> x_std;
    std::size_t input_size_;

    std::unique_ptr<wsggmlpWSGG> mlp_wsgg;

public:
    wsggMLPParameter();

    void load_mlps(const std::string& json_file);

    std::vector<double> normalize_input(const std::vector<double>& input);

    std::vector<double> get_prediction(const std::vector<double>& input);

    std::size_t input_size() const;
};

#endif


//双mlp
// #ifndef wsggMLPParameter_H
// #define wsggMLPParameter_H

// // #include "mlp.h"
// #include "wsggmlp.h"
// #include <vector>
// #include <memory>

// class wsggMLPParameter {
// public:
//     wsggMLPParameter(); // Constructor that loads MLPs from wsgg.json
//     std::vector<double> get_prediction(const std::vector<double>& input); // Get predictions

// private:
//     std::unique_ptr<wsggmlpWeights> mlp_weights; // MLP for weights prediction
//     std::unique_ptr<wsggmlpAbscs> mlp_abscs;   // MLP for absorption coefficients
//     std::vector<double> x_mean;       // Input normalization mean
//     std::vector<double> x_std;        // Input normalization std
    
//     void load_mlps(const std::string& json_file); // Load MLPs from JSON
//     std::vector<double> normalize_input(const std::vector<double>& input);
// };

// #endif // wsggMLPParameter_H
