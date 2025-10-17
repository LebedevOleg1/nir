#pragma once
#include "Mesh.hpp"

class Solver {
protected:
    Mesh* mesh;
    float alpha; // коэффициент теплопроводности (κ)
    float dt;

    virtual void calculate_dt();
    virtual void apply_bc(float* T); // применяет граничные условия к массиву T (обычно next)

public:
    Solver(Mesh* mesh, float alpha) : mesh(mesh), alpha(alpha) {
        calculate_dt();
    }

    virtual ~Solver() = default;

    virtual void initialize(float T0, float sigma);
    virtual void step() = 0;
};

// CPU-реализация решателя
class CpuSolver : public Solver {
public:
    using Solver::Solver;

    void step() override;
};
