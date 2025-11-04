#pragma once
#include "Mesh.hpp"
#include <OpenCL/opencl.h>
#include <vector>
#include <string>
#include <iostream>

class OpenCLSolver {
private:
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    // Буферы для данных на устройстве
    cl_mem bufT_curr = nullptr;
    cl_mem bufT_next = nullptr;
    cl_mem bufVolumes = nullptr;
    cl_mem bufFaces = nullptr;
    cl_mem bufCellFaces = nullptr;

    bool available = false;
    std::string kernel_src;

    // Структура для передачи данных грани в OpenCL
    struct CLFace {
        cl_int owner;
        cl_int neighbor;
        cl_float area;
        cl_float distance;
    };

public:
    OpenCLSolver() {
        build_kernel_source();
        available = init_opencl();
    }

    ~OpenCLSolver() {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (bufT_curr) clReleaseMemObject(bufT_curr);
        if (bufT_next) clReleaseMemObject(bufT_next);
        if (bufVolumes) clReleaseMemObject(bufVolumes);
        if (bufFaces) clReleaseMemObject(bufFaces);
        if (bufCellFaces) clReleaseMemObject(bufCellFaces);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    bool is_available() const { return available; }

    bool init_buffers(Mesh* mesh) {
        if (!available) return false;
        
        int_t ncells = mesh->get_ncells();
        size_t nfaces = mesh->faces.size();
        cl_int err = 0;

        // Освобождаем старые буферы
        if (bufT_curr) { clReleaseMemObject(bufT_curr); bufT_curr = nullptr; }
        if (bufT_next) { clReleaseMemObject(bufT_next); bufT_next = nullptr; }
        if (bufVolumes) { clReleaseMemObject(bufVolumes); bufVolumes = nullptr; }
        if (bufFaces) { clReleaseMemObject(bufFaces); bufFaces = nullptr; }
        if (bufCellFaces) { clReleaseMemObject(bufCellFaces); bufCellFaces = nullptr; }

        // Создаем буферы температур
        bufT_curr = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float_t) * ncells, mesh->data.curr.T.data(), &err);
        if (err != CL_SUCCESS) return false;

        bufT_next = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float_t) * ncells, nullptr, &err);
        if (err != CL_SUCCESS) return false;

        // Буфер объемов
        bufVolumes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float_t) * ncells, mesh->volumes.data(), &err);
        if (err != CL_SUCCESS) return false;

        // Подготовка данных граней для GPU
        std::vector<CLFace> cl_faces(nfaces);
        for (size_t i = 0; i < nfaces; ++i) {
            cl_faces[i].owner = mesh->faces[i].owner;
            cl_faces[i].neighbor = mesh->faces[i].neighbor;
            cl_faces[i].area = mesh->faces[i].area;
            cl_faces[i].distance = mesh->faces[i].distance;
        }

        bufFaces = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(CLFace) * nfaces, cl_faces.data(), &err);
        if (err != CL_SUCCESS) return false;

        bufCellFaces = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(cl_int) * mesh->cell_faces.size(), 
                                      mesh->cell_faces.data(), &err);
        if (err != CL_SUCCESS) return false;

        return true;
    }

    bool step_device(int_t ncells, float_t alpha, float_t dt) {
        if (!available || !kernel) return false;
        cl_int err = CL_SUCCESS;

        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufT_curr);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufT_next);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufVolumes);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufFaces);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufCellFaces);
        err |= clSetKernelArg(kernel, 5, sizeof(cl_int), &ncells);
        err |= clSetKernelArg(kernel, 6, sizeof(cl_float), &alpha);
        err |= clSetKernelArg(kernel, 7, sizeof(cl_float), &dt);

        if (err != CL_SUCCESS) return false;

        size_t global = static_cast<size_t>(ncells);
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) return false;

        clFinish(queue);
        std::swap(bufT_curr, bufT_next);
        return true;
    }

    bool read_current_to_host(Mesh* mesh) {
        if (!available) return false;
        int_t ncells = mesh->get_ncells();
        cl_int err = clEnqueueReadBuffer(queue, bufT_curr, CL_TRUE, 0, 
                                         sizeof(float_t)*ncells, 
                                         mesh->data.curr.T.data(), 
                                         0, nullptr, nullptr);
        return (err == CL_SUCCESS);
    }

private:
    void build_kernel_source() {
        kernel_src = R"CLC(
typedef struct {
    int owner;
    int neighbor;
    float area;
    float distance;
} Face;

__kernel void heat_fvm(
    __global const float* T_curr,
    __global float* T_next,
    __global const float* volumes,
    __global const Face* faces,
    __global const int* cell_faces,
    const int ncells,
    const float alpha,
    const float dt)
{
    int cell = get_global_id(0);
    if (cell >= ncells) return;
    
    float flux_sum = 0.0f;
    
    // Проходим по 4 граням ячейки
    for (int k = 0; k < 4; ++k) {
        int face_idx = cell_faces[cell * 4 + k];
        Face face = faces[face_idx];
        
        float T_owner = T_curr[face.owner];
        float T_neighbor = T_curr[face.neighbor];
        
        // Поток через грань (закон Фурье)
        float grad_T = (T_neighbor - T_owner) / face.distance;
        float flux = alpha * face.area * grad_T;
        
        flux_sum += flux;
    }
    
    // Обновление температуры
    T_next[cell] = T_curr[cell] + dt * flux_sum / volumes[cell];
}
)CLC";
    }

    bool init_opencl() {
        cl_int err;
        cl_uint num_platforms = 0;
        err = clGetPlatformIDs(0, nullptr, &num_platforms);
        if (err != CL_SUCCESS || num_platforms == 0) return false;

        std::vector<cl_platform_id> platforms(num_platforms);
        err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        if (err != CL_SUCCESS) return false;

        for (auto &p : platforms) {
            cl_uint num_devices = 0;
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) == CL_SUCCESS && num_devices > 0) {
                std::vector<cl_device_id> devices(num_devices);
                clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
                platform = p;
                device = devices[0];
                break;
            }
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices) == CL_SUCCESS && num_devices > 0) {
                std::vector<cl_device_id> devices(num_devices);
                clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, num_devices, devices.data(), nullptr);
                platform = p;
                device = devices[0];
                break;
            }
        }

        if (!device) return false;

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) return false;

        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) return false;

        const char* src = kernel_src.c_str();
        size_t src_len = kernel_src.size();
        program = clCreateProgramWithSource(context, 1, &src, &src_len, &err);
        if (err != CL_SUCCESS) return false;

        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            std::cerr << "Build failed:\n" << log << "\n";
            return false;
        }

        kernel = clCreateKernel(program, "heat_fvm", &err);
        if (err != CL_SUCCESS) return false;

        return true;
    }
};
