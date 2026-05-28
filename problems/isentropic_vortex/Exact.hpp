#pragma once
#include "Physics.hpp"
#include <cmath>
#include <vector>

// ============================================================================
// Exact solution for the isentropic vortex.
//
// At time t: vortex center is at (x0 + u_inf*t, y0 + v_inf*t) mod [Lx, Ly].
// Each primitive variable at (x,y,t) is the same function of (x - xc(t), y - yc(t)).
//
//   du = -epsilon/(2*pi) * dy * exp((1-r^2)/2)
//   dv =  epsilon/(2*pi) * dx * exp((1-r^2)/2)
//   dT = -(gamma-1)*epsilon^2/(8*gamma*pi^2) * exp(1-r^2)
//
//   rho = (T_inf + dT)^(1/(gamma-1))
//   u   = u_inf + du
//   v   = v_inf + dv
//   p   = rho^gamma / gamma   (isentropic: p/rho^gamma = const)
// ============================================================================
namespace VortexExact {
    // Compute L2 error of density field vs exact solution.
    // state: SoA [rho, rho*u, rho*v, E] with ncells_total stride.
    static inline double l2_error_rho(
        const float* state_curr,
        int nx, int ny, int ncells_total,
        float xmin, float xmax, float ymin, float ymax,
        float t_final)
    {
        using namespace std;
        const float gamma   = VortexPhysics::gamma;
        const float u_inf   = VortexPhysics::u_inf;
        const float v_inf   = VortexPhysics::v_inf;
        const float T_inf   = VortexPhysics::T_inf;
        const float eps     = VortexPhysics::epsilon;
        const float Lx      = xmax - xmin;
        const float Ly      = ymax - ymin;

        float hx = Lx / nx, hy = Ly / ny;

        // Vortex center at t_final (wrapped to [-Lx/2, Lx/2] relative to x0=0)
        float xc = VortexPhysics::x0 + u_inf * t_final;
        float yc = VortexPhysics::y0 + v_inf * t_final;
        // Wrap to domain center
        xc = fmodf(xc - xmin, Lx) + xmin;
        yc = fmodf(yc - ymin, Ly) + ymin;

        double sum2 = 0.0;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int c = i + j * nx;
                float rho_num = state_curr[0 * ncells_total + c];

                // Cell center
                float x = xmin + (i + 0.5f) * hx;
                float y = ymin + (j + 0.5f) * hy;

                // Minimum-image distance (periodic domain)
                float dx = x - xc;
                float dy = y - yc;
                // Wrap dx/dy to [-L/2, L/2]
                dx -= Lx * floorf(dx / Lx + 0.5f);
                dy -= Ly * floorf(dy / Ly + 0.5f);
                float r2 = dx * dx + dy * dy;

                float dT    = -(gamma - 1.0f) * eps * eps /
                               (8.0f * gamma * 3.14159265f * 3.14159265f) * expf(1.0f - r2);
                float T     = T_inf + dT;
                float rho_ex = powf(T / T_inf, 1.0f / (gamma - 1.0f)) * VortexPhysics::rho_inf;

                float err = rho_num - rho_ex;
                sum2 += (double)(err * err) * (double)(hx * hy);
            }
        }
        return sqrt(sum2);
    }
} // namespace VortexExact
