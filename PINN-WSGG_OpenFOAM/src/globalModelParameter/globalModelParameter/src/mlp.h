#ifndef MLP_H
#define MLP_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "json.hpp"

using json = nlohmann::json;
using std::vector;

class MLP {
protected:
    struct Layer {
        vector<vector<double>> weights;
        vector<double> bias;
        std::string activation;
        int units;
    };

    vector<Layer> layers;

    // Activation function
    inline double relu(double x) {
        return x > 0 ? x : 0;
    };

    // Forward pass
    vector<double> forward(const vector<double>& input);

public:
    MLP(const std::vector<Layer>& layers);

    // Virtual predict function
    virtual vector<double> predict(const vector<double>& input) {
        return forward(input);
    }

    virtual ~MLP() = default;
};

#endif
