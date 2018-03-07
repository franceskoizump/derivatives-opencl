#include <iostream>
#include "CL/cl.hpp"
#include <vector>
#include <complex>
#include <fstream>
#include <streambuf>

#define SIZE 20

int main()
{
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
    for (int i = 0; i < all_platform.size(); i++) 
        std::cout << all_platform[i].getInfo<CL_PLATFORM_NAME>();
    std::cout << std::endl;
    cl::Platform default_platform = all_platform[0];
    

    std::vector<cl::Device> devices;
    default_platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
   
    cl::Device default_device = devices[0];
    size_t max_x = 
                    default_device.getInfo < CL_DEVICE_MAX_WORK_GROUP_SIZE >();
    std::cout << max_x << std::endl;
    size_t max_mem = default_device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    
    cl::Context context({default_device});
    cl::Program::Sources sources;
    std::ifstream codecl("code1.cl");
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
                 CL_MEM_READ_WRITE, CL_MEM_USE_HOST_PTR);
    cl::Buffer res(context, r,    r+SIZE*SIZE,
                 CL_MEM_READ_WRITE, CL_MEM_USE_HOST_PTR);

    kernel_add.setArg(0, in);
    kernel_add.setArg(1, res);
    kernel_add.setArg(2, max_mem, NULL);

    queue.enqueueNDRangeKernel(kernel_add,
                               cl::NullRange,
                               cl::NDRange(SIZE, SIZE),
                               cl::NullRange);


    queue.finish();

    cl::copy(res, r, r+SIZE*SIZE);
    

    float max = -2;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j <SIZE; j++)
        {
            int id = i * SIZE + j;
            max = std::max(max, 
                std::abs((float)cos(i / (float)SIZE + j / (float)SIZE) - r[id]));
        }

std::cout << std::endl
          << max << std::endl
          << std::endl;

return 0;
}