// solver_cpu_stub.cpp — CPU-only replacement for solver.cu.
//
// In CUDA builds, solver.cu handles everything (it includes solver_impl.inl
// and provides the GPU specializations + explicit instantiations).
//
// In CPU-only builds (FVM_CPU_ONLY=1), this file:
//   - includes solver_impl.inl (all CPU method bodies)
//   - provides no-op GPU method specializations
//   - provides explicit template instantiations

#ifdef FVM_CPU_ONLY

#include "Advance/solver_impl.inl"

// No-op GPU steps (CPU-only: use_gpu is always false, these are never called)
template<> void Solver<PhysicsType::Heat>::step_gpu()      {}
template<> void Solver<PhysicsType::Diffusion>::step_gpu() {}
template<> void Solver<PhysicsType::Euler>::step_gpu()     {}

template<> float Solver<PhysicsType::Heat>::compute_dt_gpu()      { return compute_dt(); }
template<> float Solver<PhysicsType::Diffusion>::compute_dt_gpu() { return compute_dt(); }
template<> float Solver<PhysicsType::Euler>::compute_dt_gpu()     { return compute_dt(); }

template<> void Solver<PhysicsType::Heat>::apply_bcs_gpu()      {}
template<> void Solver<PhysicsType::Diffusion>::apply_bcs_gpu() {}
template<> void Solver<PhysicsType::Euler>::apply_bcs_gpu()     {}

// Explicit instantiations
template class Solver<PhysicsType::Heat>;
template class Solver<PhysicsType::Diffusion>;
template class Solver<PhysicsType::Euler>;

#endif  // FVM_CPU_ONLY
