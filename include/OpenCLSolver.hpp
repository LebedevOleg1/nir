#pragma once
#include "Mesh.hpp"
#include <OpenCL/opencl.h>
#include <vector>
#include <string>
#include <iostream>

class OpenCLSolver {
private:
    // OpenCL objects
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    cl_mem bufA = nullptr; // device buffer for current
    cl_mem bufB = nullptr; // device buffer for next

    bool available = false;

    std::string kernel_src;

public:
    OpenCLSolver() {
        build_kernel_source();
        available = init_opencl();
    }

    ~OpenCLSolver() {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (bufA) clReleaseMemObject(bufA);
        if (bufB) clReleaseMemObject(bufB);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    bool is_available() const { return available; }

    // Инициализация GPU буферов (копирование начального состояния curr)
    bool init_buffers(Mesh* mesh) {
        if (!available) return false;
        int_t nx = mesh->get_nx();
        int_t ny = mesh->get_ny();
        size_t N = static_cast<size_t>(nx) * static_cast<size_t>(ny);
        cl_int err = 0;

        // Если уже созданы — освободить
        if (bufA) { clReleaseMemObject(bufA); bufA = nullptr; }
        if (bufB) { clReleaseMemObject(bufB); bufB = nullptr; }

        // создаём буферы и копируем curr в bufA
        bufA = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(float_t) * N, mesh->data.curr.T.data(), &err);
        if (err != CL_SUCCESS) { std::cerr << "clCreateBuffer bufA failed: " << err << "\n"; return false; }

        // bufB инициализируем нулями (или копируем curr)
        std::vector<float_t> zeros(N, static_cast<float_t>(0));
        bufB = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(float_t) * N, zeros.data(), &err);
        if (err != CL_SUCCESS) { std::cerr << "clCreateBuffer bufB failed: " << err << "\n"; return false; }

        return true;
    }

    // Запуск одного шага (bufA -> bufB), затем swap буферов внутри класса
    bool step_device(int_t nx, int_t ny, float_t hx, float_t hy, float_t alpha, float_t dt) {
        if (!available || !kernel) return false;
        cl_int err = CL_SUCCESS;

        // Устанавливаем аргументы:
        // arg0: bufA (current)
        // arg1: bufB (next)
        // arg2: nx
        // arg3: ny
        // arg4: hx
        // arg5: hy
        // arg6: alpha
        // arg7: dt
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_int), &nx);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &ny);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_float), &hx);
        err |= clSetKernelArg(kernel, 5, sizeof(cl_float), &hy);
        err |= clSetKernelArg(kernel, 6, sizeof(cl_float), &alpha);
        err |= clSetKernelArg(kernel, 7, sizeof(cl_float), &dt);

        if (err != CL_SUCCESS) { std::cerr << "clSetKernelArg failed: " << err << "\n"; return false; }

        size_t global = static_cast<size_t>(nx) * static_cast<size_t>(ny);
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { std::cerr << "clEnqueueNDRangeKernel failed: " << err << "\n"; return false; }

        // дождёмся окончания
        clFinish(queue);

        // swap device buffers pointers
        std::swap(bufA, bufB);
        return true;
    }

    // Считать с устройства текущий буфер (bufA — всегда текущий после swap)
    bool read_current_to_host(Mesh* mesh) {
        if (!available) return false;
        int_t nx = mesh->get_nx();
        int_t ny = mesh->get_ny();
        size_t N = static_cast<size_t>(nx) * static_cast<size_t>(ny);
        cl_int err = clEnqueueReadBuffer(queue, bufA, CL_TRUE, 0, sizeof(float_t)*N, mesh->data.curr.T.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { std::cerr << "clEnqueueReadBuffer failed: " << err << "\n"; return false; }
        return true;
    }

private:
    void build_kernel_source() {
        // Явная схема с периодическими граничными условиями
        kernel_src =
R"CLC(
__kernel void heat_step(
    __global const float* Tcurr,
    __global float* Tnext,
    const int nx,
    const int ny,
    const float hx,
    const float hy,
    const float alpha,
    const float dt)
{
    int gid = get_global_id(0);
    int N = nx * ny;
    if (gid >= N) return;
    int i = gid % nx;
    int j = gid / nx;

    // Periodic neighbor indices
    int ip = (i + 1) % nx;
    int im = (i - 1 + nx) % nx;
    int jp = (j + 1) % ny;
    int jm = (j - 1 + ny) % ny;

    int id   = j * nx + i;
    int id_ip = j * nx + ip;
    int id_im = j * nx + im;
    int id_jp = jp * nx + i;
    int id_jm = jm * nx + i;

    float T0  = Tcurr[id];
    float Txp = Tcurr[id_ip];
    float Txm = Tcurr[id_im];
    float Typ = Tcurr[id_jp];
    float Tym = Tcurr[id_jm];

    float lap_x = (Txp - 2.0f * T0 + Txm) / (hx * hx);
    float lap_y = (Typ - 2.0f * T0 + Tym) / (hy * hy);

    Tnext[id] = T0 + alpha * dt * (lap_x + lap_y);
}
)CLC";
    }

    bool init_opencl() {
        cl_int err;
        cl_uint num_platforms = 0;
        err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS || num_platforms == 0) {
            std::cerr << "No OpenCL platforms found or clGetPlatformIDs failed (" << err << ")\n";
            return false;
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS) { std::cerr << "clGetPlatformIDs failed: " << err << "\n"; return false; }

        // выбираем первую платформу с устройством GPU/CPU
        for (auto &p : platforms) {
            cl_uint num_devices = 0;
            // сначала пробуем GPU
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) == CL_SUCCESS && num_devices > 0) {
                std::vector<cl_device_id> devices(num_devices);
                clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
                platform = p;
                device = devices[0];
                break;
            }
            // иначе пробуем CPU
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices) == CL_SUCCESS && num_devices > 0) {
                std::vector<cl_device_id> devices(num_devices);
                clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, num_devices, devices.data(), nullptr);
                platform = p;
                device = devices[0];
                break;
            }
        }

        if (!device) {
            std::cerr << "No suitable OpenCL device found\n";
            return false;
        }

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) { std::cerr << "clCreateContext failed: " << err << "\n"; return false; }

        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) { std::cerr << "clCreateCommandQueue failed: " << err << "\n"; return false; }

        const char* src = kernel_src.c_str();
        size_t src_len = kernel_src.size();
        program = clCreateProgramWithSource(context, 1, &src, &src_len, &err);
        if (err != CL_SUCCESS) { std::cerr << "clCreateProgramWithSource failed: " << err << "\n"; return false; }

        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            // печать логов сборки
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            std::cerr << "clBuildProgram failed: " << err << "\nBuild log:\n" << log << "\n";
            return false;
        }

        kernel = clCreateKernel(program, "heat_step", &err);
        if (err != CL_SUCCESS) { std::cerr << "clCreateKernel failed: " << err << "\n"; return false; }

        return true;
    }
};
