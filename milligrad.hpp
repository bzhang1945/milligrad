/*
 * milligrad.hpp
 * 
 * Benson Zhang
 */

#ifndef MILLIGRAD_HPP
#define MILLIGRAD_HPP

#include <memory>
#include <functional>

class Var: public std::enable_shared_from_this<Var> {
    using VarPtr = std::shared_ptr<Var>;

    public:
        double val;
        double grad;
        VarPtr prev1, prev2;
        std::function<void()> back;
        bool visited = false;

        Var(double value): Var(value, nullptr, nullptr) {}

        Var(double value, VarPtr p1, VarPtr p2): val(value), grad(0.0), prev1(p1), prev2(p2), back([]() {}) {}

        // Initiate back propagation and adjust all gradients starting at current Var.
        void backward();

    // friend ostream& operator<<(ostream& os, const Var& var) {
    // }
};

using VarPtr = std::shared_ptr<Var>;

/* Operations */
VarPtr pow(const VarPtr& a, double exp);
VarPtr pow(double a, const VarPtr& exp);

VarPtr operator+(const VarPtr& a, const VarPtr& b);
VarPtr operator+(const VarPtr& a, double b);
VarPtr operator+(double a, const VarPtr& b);

VarPtr operator*(const VarPtr& a, const VarPtr& b);
VarPtr operator*(const VarPtr& a, double b);
VarPtr operator*(double a, const VarPtr& b);

template<typename T1, typename T2>
VarPtr operator-(T1 a, T2 b) { return a + (-1 * b); };
template<typename T1, typename T2>
VarPtr operator/(T1 a, T2 b) { return a * pow(b, -1); };

VarPtr log(const VarPtr& a, double base);

VarPtr sin(const VarPtr& a);
VarPtr cos(const VarPtr& a);
VarPtr tan(const VarPtr& a);

VarPtr relu(const VarPtr& a);
VarPtr tanh(const VarPtr& a);

#endif // MILLIGRAD_HPP