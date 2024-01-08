/*
 * net.hpp - Vanilla Neural Network
 * Implements a basic multilayered perceptron with backpropagation powered by Milligrad.
 * A network comprises layers, which comprises nodes containing weights and biases.
 * Benson Zhang
 */
#ifndef NET_HPP
#define NET_HPP

#include <vector>
#include <memory>
#include <random>
#include "milligrad.hpp"

using VarPtr = std::shared_ptr<Var>;

class Module {
    public:
        void zero_grad() {
            for (VarPtr p : params()) p->grad = 0;
        }

        virtual std::vector<VarPtr> params() { return {}; }
};

class Net: public Module {
    public:
        class Node: public Module {
            private:
                std::vector<VarPtr> w;
                VarPtr b;

            public:
                Node(int inputs, std::mt19937& rng);
                VarPtr operator()(const std::vector<VarPtr>& x, bool activation);
                std::vector<VarPtr> params();
        };

        class Layer: public Module {
            private:
                std::vector<Node> nodes;
            public:
                Layer(int inputs, int outputs, std::mt19937& rng);
                std::vector<VarPtr> operator()(const std::vector<VarPtr>& x, bool activation);
                std::vector<VarPtr> params();
        };
        Net(int inputs, std::vector<int> outputs);
        std::vector<VarPtr> operator()(std::vector<VarPtr> x);
        std::vector<VarPtr> params();

    private:
        std::vector<Layer> layers;
        std::mt19937 rng;
};

// Calculates and returns mean-squared loss (L = (Y - \bar{Y})^2 / |Y|) with mini-batch estimate.
VarPtr mse_loss(const std::vector<VarPtr>& ytrue, const std::vector<VarPtr>& ypred, int batch_size, std::mt19937& rng);

// Conducts forward and backward (propagation and gradient descent) passes of a set of data for a number of epochs.
// Performs gradient descent by default; specifying batch size (1 <= batch_size <= |Y|) performs mini-batch SGD.
void train(Net& model, const std::vector<std::vector<VarPtr>>& x, const std::vector<VarPtr>& y, int epochs, double lr, int batch_size = 0);

#endif // NET_HPP
