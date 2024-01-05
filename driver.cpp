/*
 * driver.cpp - Driver class including a basic initialisation of a regression task that trains the neural network
 * defined in NET.HPP, as well as an analysis of computation time.
 */

#include <vector>
#include <cmath>
#include <iostream>
#include "net.hpp"
#include<chrono>

/*VarPtr initialisation helpers*/
// Turns a matrix (2d) of floats into a matrix of Vars.
std::vector<std::vector<VarPtr>> mat_to_Var(const std::vector<std::vector<double>>& matrix) {
    std::vector<std::vector<VarPtr>> out;
    out.reserve(matrix.size());
    for (int i = 0; i < matrix.size(); ++i) {
        std::vector<VarPtr> x;
        x.reserve(matrix[0].size());
        for (int j = 0; j < matrix[0].size(); j++) {
            x.emplace_back(std::make_shared<Var>(matrix[i][j]));
        }
        out.push_back(x);
    }
    return out;
}

// Turns an array (1d) of floats into an array of Vars.
std::vector<VarPtr> arr_to_Var(const std::vector<double>& array) {
    std::vector<VarPtr> out;
    out.reserve(array.size());
    for (int i = 0; i < array.size(); ++i) {
        out.emplace_back(std::make_shared<Var>(array[i]));
    }
    return out;
}


int main() {
    std::vector<std::vector<double>> inputs = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };

    std::vector<double> labels = {1.0, -1.0, -1.0, 1.0};

    // convert X and Y to VarPtr
    std::vector<std::vector<VarPtr>> X = mat_to_Var(inputs);
    std::vector<VarPtr> Y = arr_to_Var(labels);

    auto start = std::chrono::high_resolution_clock::now();
    // Create model: initialises a neural network with 2 hidden layers of 4 nodes each.
    // Singleton-node output layer to perform regression. 
    std::vector<int> layer_sizes = {4, 4, 1};
    Net model = Net(X[0].size(), layer_sizes);

    // train the network for 100 epochs at a learning rate at 0.05.
    train(model, X, Y, 100, 0.05);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Create and train model elapsed time: " << elapsed.count() << " seconds" << std::endl;
}