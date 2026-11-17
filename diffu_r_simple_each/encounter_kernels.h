#ifndef MFRT_ENCOUNTER_KERNELS_H
#define MFRT_ENCOUNTER_KERNELS_H
#include <cmath>
#include <stdexcept>

// Exact isotropic angular average of b^2/(b^2+epsilon^2)^2, b=r sin(theta).
// Units: inverse length squared. The small-a series avoids cancellation.
inline double position_kernel(double r_squared, double epsilon_squared) {
    if (r_squared == 0.0) return 0.0;
    const double a = r_squared / epsilon_squared;
    if (a >= 1.e4) {
        // Large-a expansion of the SAME angle average through (1/a)^3.
        // At the join the omitted relative term is < 2e-16; retain ln(a).
        // Avoid asinh and three square roots for the usual well-separated pairs.
        const double t = 1.0/a, L = 0.5*std::log(a) + 0.6931471805599453094;
        return (L-0.5 + t*(0.75-L + t*(1.125*L-0.84375
                    + t*(89.0/96.0-1.25*L)))) / r_squared;
    }
    if (a < 0.01) {
        double moment = 1.0, power = 1.0, sum = 0.0;
        for (int k=1; k<=10; ++k) {
            moment *= (2.0*k)/(2.0*k+1.0);
            power *= a;
            sum += (k%2 ? 1.0 : -1.0)*k*power*moment;
        }
        return sum/epsilon_squared;
    }
    const double x = std::sqrt(a);
    const double j = std::asinh(x)/(x*std::sqrt(1.0+a));
    return ((1.0+2.0*a)*j-1.0)/(2.0*(1.0+a)*epsilon_squared);
}

// Symmetric velocity tensor kernel; diagonal terms avoid subtraction loss.
// For epsilon=0, a distinct pair of exactly equal velocities is undefined.
template <class Tensor>
inline void add_velocity_kernel(Tensor& t, double x, double y, double z,
                                double epsilon_squared, double weight=1.0) {
    const double u2=x*x+y*y+z*z, softened=u2+epsilon_squared;
    if (softened == 0.0) throw std::domain_error("Coincident velocities with zero softening");
    const double factor=weight/(softened*std::sqrt(softened));
    t.D[0][0] += factor*(y*y+z*z);
    t.D[1][1] += factor*(z*z+x*x);
    t.D[2][2] += factor*(x*x+y*y);
    t.D[0][1] -= factor*x*y;
    t.D[0][2] -= factor*x*z;
    t.D[1][2] -= factor*y*z;
}
#endif
