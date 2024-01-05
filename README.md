# Milligrad

A lightweight yet comprehensive implementation of a scalar automatic differentiation engine, built from scratch in vanilla C++.

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).


Milligrad builds a computational DAG that enables efficient backpropagation through recursive topological sort. The system supports all basic elementary functions, including arithmetic operators, exponentiation, logarithms, and trig functions. The engine additionally implements the ReLU and tanh functions for deep learning convenience.

A vanilla Neural Network framework `net` is also implemented that runs on Milligrad to efficiently perform backpropagation and gradient descent. `net` is structured similar to PyTorch's API to enable network and training parameter customisation. By default, `net` utilises the [He Initialisation](https://paperswithcode.com/method/he-initialization) and tanh activation due to its performance on small scale regression tasks, the target for networks on scalar engines.

To run the repo, simply run `./demo`, which executes `driver.cpp`.
To compile the source code, run `g++ -o {name} milligrad.cpp net.cpp driver.cpp`.

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

We can define a 2-hidden layer NN, each with 4 nodes, in `net` and train for 50 epochs with $\alpha = 0.05$:
```
iteration: 5, loss: 0.015055
Predictions: 1.01382 -0.958675 -0.962142 0.891728 
iteration: 10, loss: 0.00695365
Predictions: 1.05742 -0.995382 -0.999699 0.939704

...

iteration: 35, loss: 0.00421941
Predictions: 1.04554 -0.998547 -0.999396 0.953705 
iteration: 40, loss: 0.00386348
Predictions: 1.04357 -0.998765 -0.999335 0.955692 
iteration: 45, loss: 0.00354656
Predictions: 1.04174 -0.998939 -0.999301 0.95754 
iteration: 50, loss: 0.00326245
Predictions: 1.04003 -0.99908 -0.999286 0.959269 
Create and train model elapsed time: 0.043667 seconds
```

### Performance
Running several iterations of a `2000`-epoch training cycle at $\alpha = 0.01$ benchmark on the above task, Milligrad and `net` averaged around ``1.00 s`` in runtime, while Karpathy's micrograd averaged around `3.61 s`, marking nearly a 4x speed increase on small datasets due to the speed of vanilla C++ and optimisations in Milligrad's DAG.

For educational purposes only.