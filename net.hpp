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
                Node(int inputs);
                VarPtr operator()(const std::vector<VarPtr>& x, bool activation);
                std::vector<VarPtr> params();
        };

        class Layer: public Module {
            private:
                std::vector<Node> nodes;
            public:
                Layer(int inputs, int outputs);
                std::vector<VarPtr> operator()(const std::vector<VarPtr>& x, bool activation);
                std::vector<VarPtr> params();
        };
        Net(int inputs, std::vector<int> outputs);
        std::vector<VarPtr> operator()(std::vector<VarPtr> x);
        std::vector<VarPtr> params();

    private:
        std::vector<Layer> layers;
};

// Mean-Squared Loss: L = (Y - \bar{Y})^2
VarPtr mse_loss(const std::vector<VarPtr>& ytrue, const std::vector<VarPtr>& ypred);

// Conduct forward and backward (propagation and gradient descent) passes of a set of data for a number of epochs.
void train(Net& model, const std::vector<std::vector<VarPtr>>& x, const std::vector<VarPtr>& y, int epochs, double lr);

#endif // NET_HPP
