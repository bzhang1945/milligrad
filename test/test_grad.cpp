/*
 *
 */
#include <memory>
#include <cmath>
#include <memory>
#include <cassert>
#include <iostream>
#include "../milligrad.hpp"

void testAddition() {
    auto a = std::make_shared<Var>(2.0);
    auto b = std::make_shared<Var>(3.0);
    auto c = a + b;

    c->backward();

    assert(c->val == 5.0); // Forward pass
    assert(a->grad == 1.0); // Backward pass
    assert(b->grad == 1.0);
}

void testSubtraction() {
    auto a = std::make_shared<Var>(10.0);
    auto b = std::make_shared<Var>(4.0);
    auto c = a - b;

    c->backward();

    assert(c->val == 6.0);
    assert(a->grad == 1.0);
    assert(b->grad == -1.0);
}

void testMultiplication() {
    auto a = std::make_shared<Var>(4.0);
    auto b = std::make_shared<Var>(5.0);
    auto c = a * b;

    c->backward();

    assert(c->val == 20.0);
    assert(a->grad == 5.0);
    assert(b->grad == 4.0);
}

void testDivision() {
    auto a = std::make_shared<Var>(10.0);
    auto b = std::make_shared<Var>(2.0);
    auto c = a / b;

    c->backward();

    assert(c->val == 5.0);
    assert(a->grad == 0.5);
    assert(b->grad == -2.5);
}

void testExponentiation() {
    auto a = std::make_shared<Var>(3.0);
    auto c = pow(a, 2);

    c->backward();

    assert(c->val == 9.0);
    assert(a->grad == 6.0);
}

void testLogarithm() {
    auto a = std::make_shared<Var>(M_E);
    auto c = log(a, M_E);

    c->backward();

    assert(c->val == 1.0);
    assert(fabs(a->grad - 1 / M_E) < 1e-6);
}

void testReLU() {
    auto a = std::make_shared<Var>(-2.0);
    auto b = std::make_shared<Var>(2.0);
    auto c = relu(a);
    auto d = relu(b);

    c->backward();
    d->backward();

    assert(c->val == 0.0);
    assert(d->val == 2.0);
    assert(a->grad == 0.0);
    assert(b->grad == 1.0);
}

void testTanh() {
    auto a = std::make_shared<Var>(1.0); // Choose a value for testing
    auto b = tanh(a);

    b->backward();

    double tanh_of_a = std::tanh(a->val); // Standard library tanh for comparison
    assert(fabs(b->val - tanh_of_a) < 1e-6); // Check forward pass

    // Check backward pass, derivative of tanh is (1 - tanh^2)
    double expected_grad = 1 - tanh_of_a * tanh_of_a;
    assert(fabs(a->grad - expected_grad) < 1e-6);
}

void Compound1() {
    auto a = std::make_shared<Var>(-3.0);
    auto b = std::make_shared<Var>(4.0);
    auto c = relu(a) * b;

    c->backward();

    assert(c->val == 0.0); // ReLU(-3) * 4 = 0
    assert(a->grad == 0.0); // Gradient should be 0 as ReLU output is 0
    assert(b->grad == 0.0); // Since ReLU output is 0, no gradient flows to b
}

void Compound2() {
    auto a = std::make_shared<Var>(-1.0);
    auto b = std::make_shared<Var>(2.0);
    auto c = std::make_shared<Var>(3.0);
    auto d = relu(a) + b * c - 5.0;

    d->backward();

    assert(d->val == 1.0); // ReLU(-1) + 2*3 - 5 = 1
    assert(a->grad == 0.0); // ReLU(-1) results in 0, no gradient flows through
    assert(b->grad == 3.0); // Gradient from multiplication operation
    assert(c->grad == 2.0); // Gradient from multiplication operation
}

void Compound3() {
    auto a = std::make_shared<Var>(-2.0);
    auto b = relu(a);
    auto c = relu(b);
    auto d = relu(c);

    d->backward();

    assert(d->val == 0.0); // ReLU applied on negative number results in 0
    assert(a->grad == 0.0); // No gradient flows back due to ReLU
    assert(b->grad == 0.0);
    assert(c->grad == 0.0);
}

int main() {
    testAddition();
    testSubtraction();
    testMultiplication();
    testDivision();
    std::cout << "Arithmetic tests passed." << std::endl;
    testExponentiation();
    testLogarithm();
    testReLU();
    testTanh();
    std::cout << "Elem function tests passed." << std::endl;
    Compound1();
    Compound2();
    Compound3();
    std::cout << "Successfully passed all tests." << std::endl;

    return 0;
}