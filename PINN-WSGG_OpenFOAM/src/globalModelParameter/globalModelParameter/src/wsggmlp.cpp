#include "wsggmlp.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

#include "json.hpp"

using json = nlohmann::json;
using std::vector;


wsggmlpWSGG::wsggmlpWSGG(const std::string& json_file)
:
    MLP(std::vector<MLP::Layer>())
{
    std::ifstream file(json_file);

    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open JSON file: " + json_file);
    }

    json j;
    file >> j;
    file.close();

    std::vector<MLP::Layer> layers;

    // ============================================================
    // New single-MLP structure
    //
    // Python:
    // self.mlp = MLP([input_dim, *hidden_dims, 17])
    //
    // JSON keys should be:
    // mlp.layers.0.weight
    // mlp.layers.0.bias
    // mlp.layers.2.weight
    // mlp.layers.2.bias
    // mlp.layers.4.weight
    // mlp.layers.4.bias
    // ...
    // ============================================================

    std::vector<int> layer_indices;

    for (int idx = 0; idx <= 20; idx += 2)
    {
        std::string bias_key =
            "mlp.layers." + std::to_string(idx) + ".bias";

        if (j.contains(bias_key) && !j[bias_key].is_null())
        {
            layer_indices.push_back(idx);
        }
    }

    if (layer_indices.empty())
    {
        throw std::runtime_error(
            "No valid layers found for single WSGG MLP in JSON. "
            "Expected keys like mlp.layers.0.bias."
        );
    }

    for (size_t i = 0; i < layer_indices.size(); ++i)
    {
        MLP::Layer layer;

        int idx = layer_indices[i];

        std::string bias_key =
            "mlp.layers." + std::to_string(idx) + ".bias";

        std::string weight_key =
            "mlp.layers." + std::to_string(idx) + ".weight";

        if (!j.contains(bias_key) || j[bias_key].is_null())
        {
            throw std::runtime_error("Missing or null key: " + bias_key);
        }

        if (!j.contains(weight_key) || j[weight_key].is_null())
        {
            throw std::runtime_error("Missing or null key: " + weight_key);
        }

        layer.bias = j[bias_key].get<vector<double>>();
        layer.units = layer.bias.size();

        auto weights = j[weight_key].get<vector<vector<double>>>();

        if (weights.empty() || weights[0].empty())
        {
            throw std::runtime_error("Empty weight matrix: " + weight_key);
        }

        // ========================================================
        // PyTorch Linear weight shape:
        // [out_features, in_features]
        //
        // This MLP implementation expects:
        // [in_features, out_features]
        //
        // Therefore transpose the matrix.
        // ========================================================

        vector<vector<double>> transposed(
            weights[0].size(),
            vector<double>(weights.size())
        );

        for (size_t r = 0; r < weights.size(); ++r)
        {
            for (size_t c = 0; c < weights[0].size(); ++c)
            {
                transposed[c][r] = weights[r][c];
            }
        }

        layer.weights = {transposed};

        // 最后一层 linear，其余层 relu
        layer.activation =
            (i == layer_indices.size() - 1) ? "linear" : "relu";

        layers.push_back(layer);
    }

    this->layers = layers;
}


vector<double> wsggmlpWSGG::predict(const vector<double>& input)
{
    return forward(input);
}
//双mlp
// #include "wsggmlp.h"
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <stdexcept>
// #include "json.hpp"

// using json = nlohmann::json;
// using std::vector;

// wsggmlpWeights::wsggmlpWeights(const std::string& json_file) : MLP(std::vector<MLP::Layer>()) {
//     std::ifstream file(json_file);
//     if (!file.is_open()) {
//         throw std::runtime_error("Unable to open JSON file.");
//     }

//     json j;
//     file >> j;

//     std::vector<MLP::Layer> layers;

//     // 自动检测实际存在的层号：0,2,4,6,...
//     std::vector<int> layer_indices;
//     for (int idx = 0; idx <= 20; idx += 2) {
//         std::string bias_key = "mlp_weights.layers." + std::to_string(idx) + ".bias";
//         if (j.contains(bias_key) && !j[bias_key].is_null()) {
//             layer_indices.push_back(idx);
//         }
//     }

//     if (layer_indices.empty()) {
//         throw std::runtime_error("No valid layers found for mlp_weights in JSON.");
//     }

//     for (size_t i = 0; i < layer_indices.size(); ++i) {
//         MLP::Layer layer;
//         int idx = layer_indices[i];

//         std::string bias_key = "mlp_weights.layers." + std::to_string(idx) + ".bias";
//         std::string weight_key = "mlp_weights.layers." + std::to_string(idx) + ".weight";

//         if (!j.contains(bias_key) || j[bias_key].is_null()) {
//             throw std::runtime_error("Missing or null key: " + bias_key);
//         }
//         if (!j.contains(weight_key) || j[weight_key].is_null()) {
//             throw std::runtime_error("Missing or null key: " + weight_key);
//         }

//         layer.bias = j[bias_key].get<vector<double>>();
//         layer.units = layer.bias.size();

//         auto weights = j[weight_key].get<vector<vector<double>>>();
//         if (weights.empty() || weights[0].empty()) {
//             throw std::runtime_error("Empty weight matrix: " + weight_key);
//         }

//         vector<vector<double>> transposed(weights[0].size(), vector<double>(weights.size()));
//         for (size_t r = 0; r < weights.size(); ++r) {
//             for (size_t c = 0; c < weights[0].size(); ++c) {
//                 transposed[c][r] = weights[r][c];
//             }
//         }
//         layer.weights = {transposed};

//         // 最后一层 linear，其余层 relu
//         layer.activation = (i == layer_indices.size() - 1) ? "linear" : "relu";

//         layers.push_back(layer);
//     }

//     this->layers = layers;
//     file.close();
// }

// wsggmlpAbscs::wsggmlpAbscs(const std::string& json_file) : MLP(std::vector<MLP::Layer>()) {
//     std::ifstream file(json_file);
//     if (!file.is_open()) {
//         throw std::runtime_error("Unable to open JSON file.");
//     }

//     json j;
//     file >> j;

//     std::vector<MLP::Layer> layers;

//     // 自动检测实际存在的层号：0,2,4,6,...
//     std::vector<int> layer_indices;
//     for (int idx = 0; idx <= 20; idx += 2) {
//         std::string bias_key = "mlp_abscs.layers." + std::to_string(idx) + ".bias";
//         if (j.contains(bias_key) && !j[bias_key].is_null()) {
//             layer_indices.push_back(idx);
//         }
//     }

//     if (layer_indices.empty()) {
//         throw std::runtime_error("No valid layers found for mlp_abscs in JSON.");
//     }

//     for (size_t i = 0; i < layer_indices.size(); ++i) {
//         MLP::Layer layer;
//         int idx = layer_indices[i];

//         std::string bias_key = "mlp_abscs.layers." + std::to_string(idx) + ".bias";
//         std::string weight_key = "mlp_abscs.layers." + std::to_string(idx) + ".weight";

//         if (!j.contains(bias_key) || j[bias_key].is_null()) {
//             throw std::runtime_error("Missing or null key: " + bias_key);
//         }
//         if (!j.contains(weight_key) || j[weight_key].is_null()) {
//             throw std::runtime_error("Missing or null key: " + weight_key);
//         }

//         layer.bias = j[bias_key].get<vector<double>>();
//         layer.units = layer.bias.size();

//         auto weights = j[weight_key].get<vector<vector<double>>>();
//         if (weights.empty() || weights[0].empty()) {
//             throw std::runtime_error("Empty weight matrix: " + weight_key);
//         }

//         vector<vector<double>> transposed(weights[0].size(), vector<double>(weights.size()));
//         for (size_t r = 0; r < weights.size(); ++r) {
//             for (size_t c = 0; c < weights[0].size(); ++c) {
//                 transposed[c][r] = weights[r][c];
//             }
//         }
//         layer.weights = {transposed};

//         // 最后一层 linear，其余层 relu
//         layer.activation = (i == layer_indices.size() - 1) ? "linear" : "relu";

//         layers.push_back(layer);
//     }

//     this->layers = layers;
//     file.close();
// }

// vector<double> wsggmlpWeights::predict(const vector<double>& input) {
//     return forward(input);
// }

// vector<double> wsggmlpAbscs::predict(const vector<double>& input) {
//     return forward(input);
// }
// #include "wsggmlp.h"
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <string>
// #include "json.hpp"

// using json = nlohmann::json;
// using std::vector;

// wsggmlpWeights::wsggmlpWeights(const std::string& json_file) : MLP(std::vector<MLP::Layer>()) {
//     std::ifstream file(json_file);
//     if (!file.is_open()) {
//         throw std::runtime_error("Unable to open JSON file.");
//     }

//     json j;
//     file >> j;

//     std::vector<MLP::Layer> layers;
//     const std::vector<int> layer_indices = {0, 2, 4}; // Layer indices in JSON
//     const std::vector<std::string> activations = {"relu", "relu", "linear"};

//     for (size_t i = 0; i < layer_indices.size(); ++i) {
//         MLP::Layer layer;
//         layer.bias = j["mlp_weights.layers." + std::to_string(layer_indices[i]) + ".bias"].get<vector<double>>();
//         layer.units = layer.bias.size(); // Units determined from bias size
        
//         // Get weights and transpose them
//         auto weights = j["mlp_weights.layers." + std::to_string(layer_indices[i]) + ".weight"].get<vector<vector<double>>>();
//         vector<vector<double>> transposed(weights[0].size(), vector<double>(weights.size()));
//         for (size_t r = 0; r < weights.size(); ++r) {
//             for (size_t c = 0; c < weights[0].size(); ++c) {
//                 transposed[c][r] = weights[r][c];
//             }
//         }
//         layer.weights = {transposed};
        
//         layer.activation = activations[i];
//         layers.push_back(layer);
//     }

//     this->layers = layers;
//     file.close();
// }

// wsggmlpAbscs::wsggmlpAbscs(const std::string& json_file) : MLP(std::vector<MLP::Layer>()) {
//     std::ifstream file(json_file);
//     if (!file.is_open()) {
//         throw std::runtime_error("Unable to open JSON file.");
//     }

//     json j;
//     file >> j;

//     std::vector<MLP::Layer> layers;
//     const std::vector<int> layer_indices = {0, 2, 4}; // Layer indices in JSON
//     const std::vector<std::string> activations = {"relu", "relu", "linear"};

//     for (size_t i = 0; i < layer_indices.size(); ++i) {
//         MLP::Layer layer;
//         layer.bias = j["mlp_abscs.layers." + std::to_string(layer_indices[i]) + ".bias"].get<vector<double>>();
//         layer.units = layer.bias.size(); // Units determined from bias size
        
//         // Get weights and transpose them
//         auto weights = j["mlp_abscs.layers." + std::to_string(layer_indices[i]) + ".weight"].get<vector<vector<double>>>();
//         vector<vector<double>> transposed(weights[0].size(), vector<double>(weights.size()));
//         for (size_t r = 0; r < weights.size(); ++r) {
//             for (size_t c = 0; c < weights[0].size(); ++c) {
//                 transposed[c][r] = weights[r][c];
//             }
//         }
//         layer.weights = {transposed};
        
//         layer.activation = activations[i];
//         layers.push_back(layer);
//     }

//     this->layers = layers;
//     file.close();
// }

// vector<double> wsggmlpWeights::predict(const vector<double>& input) {
//     return forward(input);
// }

// vector<double> wsggmlpAbscs::predict(const vector<double>& input) {
//     return forward(input);
// }
