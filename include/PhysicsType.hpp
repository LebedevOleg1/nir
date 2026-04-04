#pragma once

enum class PhysicsType { Heat, Diffusion, Euler };

template<PhysicsType P> struct PhysicsTraits;

template<> struct PhysicsTraits<PhysicsType::Heat> {
    static constexpr int NVAR = 1;
    static constexpr const char* name = "Heat";
};

template<> struct PhysicsTraits<PhysicsType::Diffusion> {
    static constexpr int NVAR = 1;
    static constexpr const char* name = "Diffusion";
};

template<> struct PhysicsTraits<PhysicsType::Euler> {
    static constexpr int NVAR = 4;
    static constexpr const char* name = "Euler";
};
