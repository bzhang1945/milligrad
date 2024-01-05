/*
 * milligrad.cpp - Milligrad
 * Builds a computational DAG optimised by only storing other Vars, and performs backprop through
 * recursive topological sort. Implements all basic elementary functions.
 * Benson Zhang
 */

#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include "milligrad.hpp"

using VarPtr = std::shared_ptr<Var>;

void Var::backward() {
    std::vector<VarPtr> topo;
    std::function<void(VarPtr)> tsort = [&](VarPtr v) {
        if (!v->visited) {
            v->visited = true;
            if (v->prev1 != nullptr) tsort(v->prev1);
            if (v->prev2 != nullptr) tsort(v->prev2);
            topo.push_back(v);
        }
    };
    
    tsort(shared_from_this());

    // backpropagation from topo sort and reset visited
    this->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); it++) {
        auto& v = *it;
        v->back();
        v->visited = false;
    }
}

/* ELEMENTARY FUNCTION IMPLEMENTATIONS*/
// EXPONENTIATION
VarPtr pow(const VarPtr& a, double exp) {
    auto out = std::make_shared<Var>(std::pow(a->val, exp));
    out->prev1 = a;
    out->back = [out, a, exp]() {
        a->grad += exp * std::pow(a->val, exp - 1) * out->grad;
    };
    return out;
}

VarPtr pow(double a, const VarPtr& exp) {
    auto out = std::make_shared<Var>(std::pow(a, exp->val));
    out->prev1 = exp;
    out->back = [out, a, exp]() {
        exp->grad += out->val * std::log(a) * out->grad;
    };
    return out;
}

// ADDITION
VarPtr operator+(const VarPtr& a, const VarPtr& b) {
    auto out = std::make_shared<Var>(a->val + b->val, a, b);
    out->back = [out, a, b]() {
        a->grad += out->grad;
        b->grad += out->grad;
    };
    return out;
}

VarPtr operator+(const VarPtr& a, double b) {
    auto out = std::make_shared<Var>(a->val + b);
    out->prev1 = a;
    out->back = [out, a]() {
        a->grad += out->grad;
    };
    return out;
}

VarPtr operator+(double a, const VarPtr& b) {
    auto out = std::make_shared<Var>(a + b->val);
    out->prev2 = b;
    out->back = [out, b]() {
        b->grad += out->grad;
    };
    return out;
}

// MULTIPLICATION
VarPtr operator*(const VarPtr& a, const VarPtr& b) {
    auto out = std::make_shared<Var>(a->val * b->val, a, b);
    out->back = [out, a, b]() {
        a->grad += b->val * out->grad;
        b->grad += a->val * out->grad;
    };
    return out;
}

VarPtr operator*(const VarPtr& a, double b) {
    auto out = std::make_shared<Var>(a->val * b);
    out->prev1 = a;
    out->back = [out, a, b]() {
        a->grad += b * out->grad;
    };
    return out;
}

VarPtr operator*(double a, const VarPtr& b) {
    auto out = std::make_shared<Var>(b->val * a);
    out->prev2 = b;
    out->back = [out, a, b]() {
        b->grad += a * out->grad;
    };
    return out;
}

// LOGARITHM
VarPtr log(const VarPtr& a, double base) {
    auto out = std::make_shared<Var>(std::log(a->val) / std::log(base));
    out->prev1 = a;
    out->back = [out, a, base]() {
        a->grad += 1 / (a->val * std::log(base)) * out->grad;
    };
    return out;
}

/* TRIGONOMETRIC FUNCTIONS */
// SINE
VarPtr sin(const VarPtr& a) {
    auto out = std::make_shared<Var>(sin(a->val));
    out->prev1 = a;
    out->back = [out, a]() {
        a->grad += cos(a->val) * out->grad;
    };
    return out;
}

// COSINE
VarPtr cos(const VarPtr& a) {
    auto out = std::make_shared<Var>(cos(a->val));
    out->prev1 = a;
    out->back = [out, a]() {
        a->grad += - 1 * cos(a->val) * out->grad;
    };
    return out;
}

// TANGENT
VarPtr tan(const VarPtr& a) {
    auto out = std::make_shared<Var>(tan(a->val));
    out->prev1 = a;
    out->back = [out, a]() {
        a->grad += pow(cos(a->val), -2) * out->grad;
    };
    return out;
}

/* NN ACTIVATION HELPER FUNCTIONS */
// RELU
VarPtr relu(const VarPtr& a) {
    double value = (a->val > 0) ? a->val : 0;
    auto out = std::make_shared<Var>(value);
    out->prev1 = a;
    out->back = [out, a]() {
        a->grad += (a->val > 0) ? out->grad : 0;
    };
    return out;
}

// TANH
VarPtr tanh(const VarPtr& a) {
    double t = (exp(2 * a->val) - 1) / (exp(2 * a->val) + 1);
    auto out = std::make_shared<Var>(t);
    out->prev1 = a;
    out->back = [out, a, t]() {
        a->grad += (1 - t * t) * out->grad;
    };
    return out;
}
