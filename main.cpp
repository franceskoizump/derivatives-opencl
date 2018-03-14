#include <iostream>
#include "stdlib.h"
#include "CL/cl.hpp"
#include <vector>
#include <complex>
#include <fstream>
#include <streambuf>
#include <chrono>


int main(int argc, char* argv[])
{
    int SIZE = 128;
    int del = 1;
    size_t x_block = 0;
    size_t y_block = 0;
    if (argc > 2 && strcmp(argv[1], "-s") == 0 ) SIZE = atoi(argv[2]);
    if (argc > 4 && strcmp(argv[3], "-d") == 0 ) del = atoi(argv[4]);
    if (argc > 6 && strcmp(argv[5], "-x") == 0 ) x_block = atoi(argv[6]);
    if (argc > 8 && strcmp(argv[7], "-y") == 0 ) y_block = atoi(argv[8]);
    float* r = new float[SIZE*SIZE]();
    
    float* grid = new float[SIZE * SIZE];
    for (int i = 0; i < SIZE; i++)
        {
            for (int j = 0; j < SIZE; j++)
            {
                grid[SIZE*i + j] = sin(i/(float)SIZE + j/(float)SIZE);
            }
        }
    std::vector<cl::Platform> all_platform;

    cl::Platform::get(&all_platform);

    cl::Platform default_platform = all_platform[0];
    

    std::vector<cl::Device> devices;
    default_platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
    cl::Device default_device = devices[0];
    size_t max_x = 
                    default_device.getInfo < CL_DEVICE_MAX_WORK_GROUP_SIZE >();
    size_t max_mem = default_device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    
    cl::Context context({default_device});
    cl::Program::Sources sources;
    std::ifstream codecl("code_dx.cl");
    std::stringstream buffer_str;
    buffer_str << codecl.rdbuf();
    std::string code = buffer_str.str();

    sources.push_back({code.c_str(), code.length()});
    

    cl::Program program(context, sources);


    if (program.build({default_device}) != CL_SUCCESS)
    {
        std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)
                  << std::endl;
    }
    cl::CommandQueue queue(context, default_device);
    cl::Kernel kernel_add = cl::Kernel(program, "func");
    cl::Buffer in(context, grid, grid+SIZE*SIZE,
                 CL_MEM_READ_ONLY, CL_MEM_USE_HOST_PTR);
    cl::Buffer res(context, r,    r+SIZE*SIZE,
                 CL_MEM_WRITE_ONLY, CL_MEM_USE_HOST_PTR);

    auto start = std::chrono::high_resolution_clock::now();
    kernel_add.setArg(0, in);
    kernel_add.setArg(1, res);
    kernel_add.setArg(2, max_mem / del, NULL);
    queue.enqueueNDRangeKernel(kernel_add,
                               cl::NullRange,
                               cl::NDRange(SIZE, SIZE),
                            (x_block == 0 || y_block == 0) ? cl::NullRange
                                              : cl::NDRange(x_block, y_block) );


    queue.finish();

    cl::copy(res, r, r+SIZE*SIZE);
    auto end = std::chrono::high_resolution_clock::now();

    float max = -2;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j <SIZE; j++)
        {
            int id = i * SIZE + j;
            max = std::max(max, 
                std::abs((float)cos(i / (float)SIZE + j / (float)SIZE) - r[id]));
        }

    std::cout << SIZE << " " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " "
              << max << std::endl;
    return 0;
}