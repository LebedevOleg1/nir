#pragma once
#include <cmath>
#include <vector>

// ============================================================================
// Analytical solution for the Sod shock tube problem.
//
// Solves iteratively for the star pressure p* and star velocity u*,
// then evaluates density and velocity at each (x, t) point.
//
// Reference: Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics",
//            Chapter 4.
// ============================================================================
struct SodExact {
    float gamma;
    float rho_L, u_L, p_L;
    float rho_R, u_R, p_R;

    SodExact(float g = 1.4f,
             float rl = 1.0f,  float ul = 0.0f, float pl = 1.0f,
             float rr = 0.125f, float ur = 0.0f, float pr = 0.1f)
        : gamma(g), rho_L(rl), u_L(ul), p_L(pl),
          rho_R(rr), u_R(ur), p_R(pr) {}

    struct State { float rho, u, p; };

    // Evaluate the exact solution at position x at time t.
    // x0 = initial discontinuity location.
    State eval(float x, float t, float x0 = 0.5f) const {
        if (t < 1e-12f) {
            if (x < x0) return {rho_L, u_L, p_L};
            else         return {rho_R, u_R, p_R};
        }

        float g   = gamma;
        float g1  = (g - 1.0f) / (2.0f * g);
        float g2  = (g + 1.0f) / (2.0f * g);
        float g3  = 2.0f * g / (g - 1.0f);
        float g4  = 2.0f / (g - 1.0f);
        float g5  = 2.0f / (g + 1.0f);
        float g6  = (g - 1.0f) / (g + 1.0f);
        float g7  = (g - 1.0f) / 2.0f;

        float cL  = std::sqrt(g * p_L / rho_L);
        float cR  = std::sqrt(g * p_R / rho_R);

        // Newton-Raphson iteration for star pressure p*
        float p_star = 0.5f * (p_L + p_R);  // initial guess
        for (int iter = 0; iter < 100; ++iter) {
            float fL, fR, dfL, dfR;
            // Left wave (rarefaction or shock)
            if (p_star <= p_L) {
                float r  = p_star / p_L;
                fL  = g4 * cL * (std::pow(r, g1) - 1.0f);
                dfL = (1.0f / (rho_L * cL)) * std::pow(r, -g2);
            } else {
                float AL = g5 / rho_L;
                float BL = g6 * p_L;
                float sqt = std::sqrt(AL / (p_star + BL));
                fL  = (p_star - p_L) * sqt;
                dfL = sqt * (1.0f - (p_star - p_L) / (2.0f * (p_star + BL)));
            }
            // Right wave (rarefaction or shock)
            if (p_star <= p_R) {
                float r  = p_star / p_R;
                fR  = g4 * cR * (std::pow(r, g1) - 1.0f);
                dfR = (1.0f / (rho_R * cR)) * std::pow(r, -g2);
            } else {
                float AR = g5 / rho_R;
                float BR = g6 * p_R;
                float sqt = std::sqrt(AR / (p_star + BR));
                fR  = (p_star - p_R) * sqt;
                dfR = sqt * (1.0f - (p_star - p_R) / (2.0f * (p_star + BR)));
            }
            float dp = -(fL + fR + u_R - u_L) / (dfL + dfR);
            p_star += dp;
            if (std::abs(dp) < 1e-6f * p_star) break;
        }
        float u_star = 0.5f * (u_L + u_R + (p_star <= p_R ?
            g4*cR*(std::pow(p_star/p_R, g1)-1.0f) :
            (p_star-p_R)*std::sqrt((g5/rho_R)/(p_star+g6*p_R)))
            - (p_star <= p_L ?
            g4*cL*(std::pow(p_star/p_L, g1)-1.0f) :
            (p_star-p_L)*std::sqrt((g5/rho_L)/(p_star+g6*p_L))));

        float xi = (x - x0) / t;  // characteristic variable

        // Left region
        if (xi <= u_L - cL) return {rho_L, u_L, p_L};

        // Left rarefaction (if p* < p_L)
        if (p_star <= p_L) {
            float cL_star = cL * std::pow(p_star/p_L, g1);
            if (xi <= u_star - cL_star) {
                // Inside rarefaction fan
                if (xi > u_L - cL) {
                    float u_fan   = g5 * (cL + g7*u_L + xi);
                    float rho_fan = rho_L * std::pow(
                        g5 + g6/cL*(u_L - xi), g4);
                    float p_fan   = p_L * std::pow(rho_fan/rho_L, g);
                    return {rho_fan, u_fan, p_fan};
                }
            }
            // Left star region
            if (xi <= u_star) {
                float rho_Ls = rho_L * std::pow(p_star/p_L, 1.0f/g);
                return {rho_Ls, u_star, p_star};
            }
        } else {
            // Left shock
            float rho_Ls = rho_L * ((g+1.0f)*p_star + (g-1.0f)*p_L) /
                                   ((g-1.0f)*p_star + (g+1.0f)*p_L);
            float S_L    = u_L - cL * std::sqrt((g+1.0f)/(2.0f*g) * p_star/p_L + g1);
            if (xi <= S_L) return {rho_L, u_L, p_L};
            if (xi <= u_star) return {rho_Ls, u_star, p_star};
        }

        // Right star region
        if (xi <= u_star) {
            float rho_Rs = (p_star <= p_R) ?
                rho_R * std::pow(p_star/p_R, 1.0f/g) :
                rho_R * ((g+1.0f)*p_star + (g-1.0f)*p_R) /
                        ((g-1.0f)*p_star + (g+1.0f)*p_R);
            return {rho_Rs, u_star, p_star};
        }

        // Right rarefaction (if p* < p_R)
        if (p_star <= p_R) {
            float cR_star = cR * std::pow(p_star/p_R, g1);
            if (xi < u_star + cR_star) {
                float u_fan   = g5 * (-cR + g7*u_R + xi);
                float rho_fan = rho_R * std::pow(
                    g5 - g6/cR*(u_R - xi), g4);
                float p_fan   = p_R * std::pow(rho_fan/rho_R, g);
                return {rho_fan, u_fan, p_fan};
            }
        } else {
            // Right shock
            float S_R = u_R + cR * std::sqrt((g+1.0f)/(2.0f*g) * p_star/p_R + g1);
            if (xi < S_R) {
                float rho_Rs = rho_R * ((g+1.0f)*p_star + (g-1.0f)*p_R) /
                                       ((g-1.0f)*p_star + (g+1.0f)*p_R);
                return {rho_Rs, u_star, p_star};
            }
        }

        // Right undisturbed region
        return {rho_R, u_R, p_R};
    }

    // Compute L2 error of density field against exact solution.
    // U: SoA array [NVAR * ncells_total], ncells real cells.
    // x_coords: cell center x positions.
    float l2_error_density(const float* U, int ncells, int ncells_total,
                            const float* x_coords, float t, float x0 = 0.5f) const {
        double sum = 0.0;
        for (int i = 0; i < ncells; ++i) {
            float rho_num = U[i];                    // var 0 = density
            float rho_ex  = eval(x_coords[i], t, x0).rho;
            double err    = rho_num - rho_ex;
            sum += err * err;
        }
        return float(std::sqrt(sum / ncells));
    }
};
