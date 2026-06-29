#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "json.hpp"
#include "mlp.h"

using json = nlohmann::json;
using std::vector;

vector<double> MLP::forward(const vector<double>& input) {
    vector<double> a = input;
    for (auto& layer : layers) {
        vector<double> z(layer.units, 0.0);
        for (int i = 0; i < layer.units; ++i) {
            for (size_t j = 0; j < a.size(); ++j) {
                // std::cout << i << " " << j << " " << layer.units << " " << layer.weights.size() << layer.bias.size() << std::endl;
                z[i] += a[j] * layer.weights[j][i];
            }
            z[i] += layer.bias[i];
            if (layer.activation == "relu") {
                z[i] = relu(z[i]);
            }
        }
        a = z;
    }
    return a;
}

MLP::MLP(const std::vector<Layer>& layers) : layers(layers) {}