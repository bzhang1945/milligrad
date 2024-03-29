/*
 * net.cpp - Vanilla Neural Network
 * Implements a basic multilayered perceptron with backpropagation powered by Milligrad.
 * Nodes utilise the He Initialisation on weights and the tanh activation function on non-output layers.
 * Training performs mini-batch SGD based on the mean squared loss function.
 * Benson Zhang
 */
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include "milligrad.hpp"
#include "net.hpp"

using VarPtr = std::shared_ptr<Var>;

Net::Node::Node(int inputs, std::mt19937& rng) {
    // Xavier Initialisation
    std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / inputs));
    for (int i = 0; i < inputs; ++i) {
        w.emplace_back(std::make_shared<Var>(dist(rng)));
    }

    b = std::make_shared<Var>(dist(rng));
}

VarPtr Net::Node::operator()(const std::vector<VarPtr>& x, bool activation) {
    // R(w^T x + b)
    VarPtr dp = std::make_shared<Var>(0.0);
    for (int i = 0; i < w.size(); i++) {
        dp = dp + w[i] * x[i];
    }
    dp = dp + b;

    if (activation) return tanh(dp);
    return dp;
}

std::vector<VarPtr> Net::Node::params() {
    std::vector<VarPtr> p = w;
    p.push_back(b);
    return p;
}

 
Net::Layer::Layer(int inputs, int outputs, std::mt19937& rng) {
    for (int i = 0; i < outputs; ++i) {
        nodes.emplace_back(Node(inputs, rng));
    }
}

std::vector<VarPtr> Net::Layer::operator()(const std::vector<VarPtr>& x, bool activation) {
    std::vector<VarPtr> layer; 
    for (auto& n : nodes) {
        layer.emplace_back(n(x, activation));
    }
    return layer;
}

std::vector<VarPtr> Net::Layer::params() {
    std::vector<VarPtr> p;
    for (auto& n : nodes) {
        std::vector<VarPtr> node_params = n.params();
        p.reserve(p.size() + std::distance(node_params.begin(), node_params.end()));
        p.insert(p.end(), node_params.begin(), node_params.end());
    }
    return p;
}


Net::Net(int inputs, std::vector<int> outputs) {
    std::random_device rd;
    rng.seed(rd());
    // outputs: vector[layer1 size, layer2 size, ..., ], outputs size must >= 1
    layers.emplace_back(Layer(inputs, outputs[0], rng));
    for (int i = 0; i < outputs.size() - 1; i++) {
        layers.emplace_back(Layer(outputs[i], outputs[i+1], rng));
    }
}

std::vector<VarPtr> Net::operator()(std::vector<VarPtr> x) {
    // forward pass
    for (int i = 0; i < layers.size() - 1; ++i) {
        x = layers[i](x, true);
    }
    x = layers[layers.size() - 1](x, false); // dont activate output layer
    return x;
}

std::vector<VarPtr> Net::params() {
    std::vector<VarPtr> p;
    for (auto& layer : layers) {
        std::vector<VarPtr> layer_params = layer.params();
        p.reserve(p.size() + std::distance(layer_params.begin(), layer_params.end()));
        p.insert(p.end(), layer_params.begin(), layer_params.end());
    }
    return p;
}

VarPtr mse_loss(const std::vector<VarPtr>& ytrue, const std::vector<VarPtr>& ypred, int batch_size, std::mt19937& rng) {
    // Create a random permutation of the size of y to simulate drawing batches
    std::vector<int> perm(ytrue.size());
    for (int i = 0; i < ytrue.size(); ++i) perm[i] = i;
    std::shuffle(std::begin(perm), std::end(perm), rng);

    auto loss = std::make_shared<Var>(0.0);
    for (int i = 0; i < batch_size; ++i) {
        int idx = perm[i];
        loss = loss + pow(ytrue[idx] - ypred[idx], 2);
    }
    return loss / batch_size;
}

void train(Net& model, const std::vector<std::vector<VarPtr>>& x, const std::vector<VarPtr>& y, int epochs, double lr, int batch_size) {
    std::mt19937 rng{std::random_device{}()};
    int bs = (batch_size == 0) ? y.size() : bs;
    
    for (int e = 1; e <= epochs; ++e) {
        // forward pass model
        std::vector<VarPtr> y_pred;
        for (int i = 0; i < x.size(); ++i) {
            // currently assumes the case with 1 output node per input
            y_pred.emplace_back(model(x[i])[0]);
        }
        // calculate loss (stochastic), flush, backprop
        auto loss = mse_loss(y, y_pred, bs, rng);
        model.zero_grad();
        loss->backward();
        
        // gradient descent updates
        std::vector<VarPtr> parameters = model.params();
        for (auto& p : parameters) {
            p->val = p->val - lr * p->grad;
        }

        if (e % 5 == 0) {
            std::cout << "iteration: " << e << ", loss: " << loss->val << std::endl;

            // print out ypred
            std::cout << "Predictions: ";
            for (int i = 0; i < y_pred.size(); i++) {
                std::cout << y_pred[i]->val << " ";
            }
            std::cout << std::endl;
        }
    }    
}
