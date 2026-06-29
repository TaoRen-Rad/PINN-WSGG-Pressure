#include "wsggMLPParameter.h"
#include "wsggmlp.h"

#include <stdexcept>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

#include "json.hpp"
#include "support_func.h"

using json = nlohmann::json;


wsggMLPParameter::wsggMLPParameter()
:
    input_size_(0),
    mlp_wsgg(nullptr)
{
    std::string dl_path = GetLibraryPath();

    std::string json_file =
        dl_path + "/../../../src/globalModelParameter/parameter/wsgg.json";

    load_mlps(json_file);
}


void wsggMLPParameter::load_mlps(const std::string& json_file)
{
    std::ifstream file(json_file);

    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open JSON file: " + json_file);
    }

    json j;
    file >> j;
    file.close();

    x_mean = j["x_mean"].get<std::vector<double>>();
    x_std  = j["x_std"].get<std::vector<double>>();

    if (x_mean.size() != x_std.size())
    {
        throw std::runtime_error(
            "x_mean and x_std must have the same size."
        );
    }

    if (x_mean.size() != 4 && x_mean.size() != 5)
    {
        throw std::runtime_error(
            "WSGG input size must be 4 or 5: [T, x_H2O, x_CO2, x_CO/NH3] or [T, x_H2O, x_CO2, x_CO/NH3, P]."
        );
    }

    input_size_ = x_mean.size();

    for (size_t i = 0; i < x_std.size(); ++i)
    {
        if (std::abs(x_std[i]) < 1.0e-14)
        {
            throw std::runtime_error("x_std contains zero or near-zero value.");
        }
    }

    // Single MLP:
    // input  : 4 or 5, depending on the deployed JSON normalization vectors
    // output : 17
    //
    // output[0:9]   -> weights_raw
    // output[9:17]  -> abscs_raw
    //
    // 注意:
    // 这里要求 wsggmlp.h / wsggmlp.cpp 中已经有 wsggmlpWSGG 类。
    mlp_wsgg = std::make_unique<wsggmlpWSGG>(json_file);
}


std::vector<double> wsggMLPParameter::normalize_input(
    const std::vector<double>& input
)
{
    if (input.size() != input_size_)
    {
        throw std::invalid_argument(
            "Input vector size does not match the deployed WSGG model."
        );
    }

    std::vector<double> normalized(input.size(), 0.0);
    std::vector<double> transformed(input);

    // Keep the deployed OpenFOAM model consistent with the Python training
    // path: the fifth input is log(p), while the physical scaling by p is
    // applied later when assembling the band absorption coefficients.
    if (input_size_ == 5)
    {
        transformed[4] = std::log(std::max(input[4], 1.0e-8));
    }

    for (size_t i = 0; i < input.size(); ++i)
    {
        normalized[i] = (transformed[i] - x_mean[i]) / x_std[i];
    }

    return normalized;
}


std::vector<double> wsggMLPParameter::get_prediction(
    const std::vector<double>& input
)
{
    if (input.size() != input_size_)
    {
        throw std::invalid_argument(
            "Input vector size does not match the deployed WSGG model."
        );
    }

    // ============================================================
    // 1. Normalize input
    // ============================================================

    std::vector<double> normalized = normalize_input(input);

    // ============================================================
    // 2. Single MLP prediction
    // ============================================================
    //
    // Python model:
    // self.mlp = MLP([input_dim, *hidden_dims, 17])
    //
    // Output:
    // raw_output[0:9]   -> weights_raw
    // raw_output[9:17]  -> abscs_raw
    // ============================================================

    std::vector<double> raw_output = mlp_wsgg->predict(normalized);

    if (raw_output.size() != 17)
    {
        throw std::runtime_error(
            "WSGG MLP output size must be 17: 9 weights + 8 absorption coefficients."
        );
    }

    // ============================================================
    // 3. Weights: softmax(raw_output[0:9])
    // ============================================================

    std::vector<double> weights(9, 0.0);

    double max_w = raw_output[0];

    for (size_t i = 1; i < 9; ++i)
    {
        if (raw_output[i] > max_w)
        {
            max_w = raw_output[i];
        }
    }

    double sum_w = 0.0;

    for (size_t i = 0; i < 9; ++i)
    {
        weights[i] = std::exp(raw_output[i] - max_w);
        sum_w += weights[i];
    }

    if (sum_w <= 0.0)
    {
        throw std::runtime_error("Softmax denominator is zero or negative.");
    }

    for (size_t i = 0; i < 9; ++i)
    {
        weights[i] /= sum_w;
    }

    // ============================================================
    // 4. Absorption coefficients
    // ============================================================
    //
    // Python:
    // abscs = torch.zeros_like(weights)
    // abscs[:, 1:] = torch.exp(abscs_raw)
    //
    // Here:
    // abscs[0] = 0.0, transparent gas
    // abscs[1]~abscs[8] = exp(raw_output[9]~raw_output[16])
    // ============================================================

    std::vector<double> abscs(9, 0.0);

    abscs[0] = 0.0;

    for (size_t i = 1; i < 9; ++i)
    {
        abscs[i] = std::exp(raw_output[8 + i]);
        // i = 1 -> raw_output[9]
        // i = 8 -> raw_output[16]
    }

    // ============================================================
    // 5. Combine result
    // ============================================================
    //
    // result[0:9]   = weights
    // result[9:18]  = abscs
    //
    // In WSGGModel.C:
    // ai_[bandI] = output[bandI]
    // ki_[bandI] = output[nBands_ + bandI] * x_sum * 100.0
    // ============================================================

    std::vector<double> result;
    result.reserve(18);

    result.insert(result.end(), weights.begin(), weights.end());
    result.insert(result.end(), abscs.begin(), abscs.end());

    return result;
}

std::size_t wsggMLPParameter::input_size() const
{
    return input_size_;
}
//双mlp
// #include "wsggMLPParameter.h"
// #include "wsggmlp.h"
// #include <stdexcept>
// #include <string>
// #include <fstream>
// #include "json.hpp"
// #include <cmath>
// #include "support_func.h"

// using json = nlohmann::json;

// wsggMLPParameter::wsggMLPParameter() {
//     std::string dl_path = GetLibraryPath();
//     std::string json_file = dl_path + "/../../../src/globalModelParameter/parameter/wsgg.json";
//     load_mlps(json_file);
// }

// void wsggMLPParameter::load_mlps(const std::string& json_file) {
//     std::ifstream file(json_file);
//     if (!file.is_open()) {
//         throw std::runtime_error("Unable to open JSON file: " + json_file);
//     }

//     json j;
//     file >> j;

//     x_mean = j["x_mean"].get<std::vector<double>>();
//     x_std = j["x_std"].get<std::vector<double>>();

//     file.close();
//     mlp_weights = std::make_unique<wsggmlpWeights>(json_file);
//     mlp_abscs = std::make_unique<wsggmlpAbscs>(json_file);
// }

// std::vector<double> wsggMLPParameter::normalize_input(const std::vector<double>& input) {
//     std::vector<double> normalized(input.size());
//     for (size_t i = 0; i < input.size(); ++i) {
//         normalized[i] = (input[i] - x_mean[i]) / x_std[i];
//     }
//     return normalized;
// }

// std::vector<double> wsggMLPParameter::get_prediction(const std::vector<double>& input) {
//     if (input.size() != 4) {
//         throw std::invalid_argument("Input vector must have size 4");
//     }

//     auto normalized = normalize_input(input);

//     // weights prediction + softmax
//     auto weights = mlp_weights->predict(normalized);
//     double sum = 0.0;
//     for (auto& w : weights) { w = exp(w); sum += w; }
//     for (auto& w : weights) { w /= sum; }

//     // abscs prediction (only last 3 inputs)
//     std::vector<double> abscs_input(normalized.begin() + 1, normalized.end());
//     auto raw_abscs = mlp_abscs->predict(abscs_input);

//     // 动态大小，band0占位为0（透明气体band）
//     std::vector<double> abscs(raw_abscs.size() + 1, 0.0);
//     for (size_t i = 1; i < abscs.size(); ++i) {
//         abscs[i] = exp(raw_abscs[i - 1]);
//     }

//     // Combine: weights(9) + abscs(9) 共18个
//     std::vector<double> result;
//     result.reserve(weights.size() + abscs.size());
//     result.insert(result.end(), weights.begin(), weights.end());
//     result.insert(result.end(), abscs.begin(), abscs.end());
//     return result;
// }
