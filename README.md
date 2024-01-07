# Milligrad

A lightweight yet comprehensive implementation of a scalar automatic differentiation engine, built from scratch in vanilla C++.

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).


Milligrad builds a computational DAG that enables efficient backpropagation through topological sort. The system supports all basic elementary functions, including arithmetic operators, exponentiation, logarithms, and trig functions. The engine additionally implements the ReLU and tanh functions for deep learning convenience.

A simple Neural Network framework `net` is also implemented that runs on Milligrad to efficiently perform backpropagation and mini-batch SGD. `net` is structured similar to PyTorch's API to enable network and training parameter customisation. By default, `net` utilises the [He Initialisation](https://paperswithcode.com/method/he-initialization) and tanh activation due to its performance on small scale regression tasks, the target for networks on scalar engines.

### Demo
Consider a very simple $\mathbb{R}^3 \rightarrow \mathbb{R}$ neural network regression task, with the following data: 
``` 
X = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    }

Y = {1.0, -1.0, -1.0, 1.0}
```

We can define a 2-hidden layer NN, each with 4 nodes, in `net` and train for 50 epochs with $\alpha = 0.1$:
```
iteration: 5, loss: 0.761261
Predictions: 0.46275 0.0854935 0.16595 0.532378 
iteration: 10, loss: 0.0327976

...

iteration: 35, loss: 0.00722901
Predictions: 1.07434 -0.969646 -0.902282 1.11366 
iteration: 40, loss: 0.00356955
Predictions: 0.910624 -1.06527 -1.0145 0.957339 
iteration: 45, loss: 0.00178555
Predictions: 1.02367 -0.991781 -0.949195 1.06272 
iteration: 50, loss: 0.000940768
Predictions: 0.94996 -1.03441 -1.00094 0.991388 
Create and train model elapsed time: 0.0273787 seconds
```

### Performance
Running several iterations of a `2000`-epoch training cycle at $\alpha = 0.01$ benchmark on the above task, Milligrad and `net` averaged around ``0.86 s`` in runtime, while Karpathy's micrograd averaged around `3.61 s`, marking over a 4x speed increase even on small datasets due to the speed of vanilla C++ and optimisations in Milligrad's DAG.

### Brief documentation
To run the repo, simply run `./demo`, which executes `driver.cpp`.
To compile the source code, run `g++ -o {name} milligrad.cpp net.cpp driver.cpp`.

To use the neural network, simply create X and Y as 2d and 1d float vectors, respectively, in `driver.cpp`, and convert them to `Var` using the provided `mat_to_Var` and `arr_to_Var` functions in order to facilitate the Milligrad engine.

`Net(int inputs, std::vector<int> outputs)` creates a network, where `outputs` denotes the number of nodes at each layer (excluding the input). The last/output layer should be 1.

`train(Net& model, const std::vector<std::vector<VarPtr>>& x, const std::vector<VarPtr>& y, int epochs, double lr, int batch_size = 0)` trains the model with inputs `x` and labels `y` for a set number of epochs at a set learning rate. If a batch size is specified, the model trains using mini-batch SGD; otherwise, the model defaults to vanilla GD.

For educational purposes only.