#pragma once

#include "model.hpp"
#include "../models/nn.hpp"


class nn_command : public model_command {
protected:
    float dropout_prob, eta, lambda;
public:
    nn_command() {
        using namespace boost::program_options;

        options_desc.add_options()
            ("dropout_prob", value<float>(&dropout_prob)->default_value(0.00), "dropout prob")
            ("eta", value<float>(&eta)->default_value(0.02), "learning rate")
            ("lambda", value<float>(&lambda)->default_value(0.00002), "l2 regularization coeff");
    }

    virtual std::string name() { return "nn"; }
    virtual std::string description() { return "train and apply nn model"; }

    virtual std::unique_ptr<model> create_model(uint32_t n_fields, uint32_t n_indices, uint32_t n_index_bits) {
        return std::unique_ptr<model>(new nn_model(n_indices, n_index_bits, seed, dropout_prob, eta, lambda));
    }
};
