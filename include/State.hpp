#pragma once
#include "Types.hpp"
#include <vector>
#include <utility>

struct Float3 {
    float_t x, y, z;
    HOST_DEVICE Float3(float_t x = 0, float_t y = 0, float_t z = 0) : x(x), y(y), z(z) {}
};

struct State {
    std::vector<float_t> T;   
    std::vector<Float3> v;    
    std::vector<float_t> P;   
    int_t size;

    State() : size(0) {}
    State(int_t size_) { resize(size_); }

    void resize(int_t s) {
        size = s;
        T.assign(size, static_cast<float_t>(0));
        v.assign(size, Float3());
        P.assign(size, static_cast<float_t>(0));
    }
};

struct ExtState {
    State curr;
    State next;

    ExtState() {}
    ExtState(int_t size) { init(size); }

    void init(int_t size) {
        curr.resize(size);
        next.resize(size);
    }

     void swap_buffers() {
        std::swap(curr.T, next.T);
    }
};