/*
 * net.cpp - Vanilla Neural Network
 * Implements a basic multilayered perceptron with backpropagation powered by Milligrad.
 * Benson Zhang
 * 
 */
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <iostream>

#include "milligrad.hpp"

using VarPtr = std::shared_ptr<Var>;
class Module {
    public:
        void zero_grad() {
            for (auto& p : params()) p->grad = 0;
        }

        virtual std::vector<VarPtr> params() {
            std::vector<VarPtr> p;
            return p;
        }
};

class Net: public Module {
    public:        
        class Node: public Module {
            private:
                std::vector<VarPtr> w;
                VarPtr b;

            public:
                Node(int inputs) {
                    std::mt19937 rng{std::random_device{}()};
                    // He Initialisation
                    std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / inputs));
                    for (int i = 0; i < inputs; ++i) {
                        w.emplace_back(std::make_shared<Var>(dist(rng)));
                    }

                    b = std::make_shared<Var>(dist(rng));
                }

                VarPtr operator()(const std::vector<VarPtr>& x, bool activation) {
                    // R(w^T x + b)
                    assert(("Dimension mismatch", x.size() == w.size()));
                    VarPtr dp = std::make_shared<Var>(0.0);
                    for (int i = 0; i < w.size(); i++) {
                        dp = dp + w[i] * x[i];
                    }
                    dp = dp + b;

                    if (activation) return tanh(dp);
                    return dp;
                }

                std::vector<VarPtr> params() {
                    std::vector<VarPtr> p = w;
                    p.push_back(b);
                    return p;
                }
        };

        class Layer: public Module {
            public:
                std::vector<Node> nodes;
                Layer(int inputs, int outputs) {
                    for (int i = 0; i < outputs; ++i) {
                        nodes.emplace_back(Node(inputs));
                    }
                }

                std::vector<VarPtr> operator()(const std::vector<VarPtr>& x, bool activation) {
                    std::vector<VarPtr> layer; 
                    for (auto& n : nodes) {
                        layer.emplace_back(n(x, activation));
                    }
                    return layer;
                }

                std::vector<VarPtr> params() {
                    std::vector<VarPtr> p;
                    for (auto& n : nodes) {
                        std::vector<VarPtr> node_params = n.params();
                        p.reserve(p.size() + std::distance(node_params.begin(), node_params.end()));
                        p.insert(p.end(), node_params.begin(), node_params.end());
                    }
                    return p;
                }
        };

        std::vector<Layer> layers;

        Net(int inputs, std::vector<int> outputs) {
            // outputs: vector[layer1 size, layer2 size, ..., ], outputs size must >= 1
            layers.emplace_back(Layer(inputs, outputs[0]));
            for (int i = 0; i < outputs.size() - 1; i++) {
                layers.emplace_back(Layer(outputs[i], outputs[i+1]));
            }
        }

        std::vector<VarPtr> operator()(std::vector<VarPtr> x) {
            // forward pass
            for (int i = 0; i < layers.size() - 1; ++i) {
                x = layers[i](x, true);
            }
            x = layers[layers.size() - 1](x, false); // dont activate output layer
            return x;
        }

        std::vector<VarPtr> params() {
            std::vector<VarPtr> p;
            for (auto& layer : layers) {
                std::vector<VarPtr> layer_params = layer.params();
                p.reserve(p.size() + std::distance(layer_params.begin(), layer_params.end()));
                p.insert(p.end(), layer_params.begin(), layer_params.end());
            }
            return p;
        }
};

VarPtr mse_loss(const std::vector<VarPtr>& ytrue, const std::vector<VarPtr>& ypred) { // , int batch_size
    assert(ytrue.size() == ypred.size());
    //if (batch_size = -1) {
    //}
    auto loss = std::make_shared<Var>(0.0);
    for (int i = 0; i < ytrue.size(); ++i) {
        loss = loss + pow(ytrue[i] - ypred[i], 2);
    }
    return loss;
}

void train(Net& model, const std::vector<std::vector<VarPtr>>& x, const std::vector<VarPtr>& y, int epochs, double lr) {

    for (int e = 1; e <= epochs; ++e) {
        // forward pass model
        std::vector<VarPtr> y_pred;
        for (int i = 0; i < x.size(); ++i) {
            // currently only supports the case with 1 output node per input
            y_pred.emplace_back(model(x[i])[0]);
        }

        // get loss, flush, backprop
        auto loss = mse_loss(y, y_pred);
        model.zero_grad();
        loss->backward();
        
        // gradient descent
        std::vector<VarPtr> parameters = model.params();
        for (auto& p : parameters) {
            p->val = p->val - lr * p->grad;
        }

        if (e % 5 == 0) {
            std::cout << "iteration: " << e << ", loss: " << loss->val << std::endl;

            // print out ypred
            for (int i = 0; i < y_pred.size(); i++) {
                std::cout << y_pred[i]->val << " ";
            }
            std::cout << std::endl;
        }
    }    
}


