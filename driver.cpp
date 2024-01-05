#include <vector>
#include <cmath>
#include <iostream>
#include "net.cpp"
#include<chrono>

int main() {
    std::vector<std::vector<double>> X = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };

    std::vector<double> Y = {1.0, -1.0, -1.0, 1.0};

    // convert X and Y to VarPtr
    std::vector<std::vector<VarPtr>> input;
    for (int i = 0; i < X.size(); ++i) {
        std::vector<VarPtr> x;
        for (int j = 0; j < X[0].size(); j++) {
            x.emplace_back(std::make_shared<Var>(X[i][j]));
        }
        input.push_back(x);
    }

    std::vector<VarPtr> label;
    for (int i = 0; i < Y.size(); ++i) {
        label.emplace_back(std::make_shared<Var>(Y[i]));
    }

    auto start = std::chrono::high_resolution_clock::now();
    // create model
    std::vector<int> layer_sizes = {4, 4, 1};
    Net model = Net(X[0].size(), layer_sizes);

    train(model, input, label, 100, 0.05);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Create and train model elapsed time: " << elapsed.count() << " seconds" << std::endl;

}