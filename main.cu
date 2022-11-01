#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Image.cuh"
#include "FileHandler.cuh"


using namespace std;
typedef std::chrono::high_resolution_clock timer;

void checkCalcError(const thrust::host_vector<pixel> cpuOut, const thrust::host_vector<pixel> gpuOut) {
    if (cpuOut.size() != gpuOut.size()) {
        cout << "Size mismatch" << endl;
        return;
    }
    double error = 0;
    for (size_t i = 0; i < cpuOut.size(); i++) {
        error += abs(cpuOut[i].r - gpuOut[i].r);
        error += abs(cpuOut[i].g - gpuOut[i].g);
        error += abs(cpuOut[i].b - gpuOut[i].b);
    }
    cout << "Total absolute error: " << error << endl;
}


void getGaussianKernel(thrust::host_vector<double>& kernel, int sigma) {
    /* This function returns a vector containing a 1d gaussian kernel assuming 0 mean.
     * Since we are assuming the kernel is symmetric, this is a more
     * efficient approach instead of using a 2d gaussian kernel.
     */
    int k = 6 * sigma + 1;
    double sig = (double) sigma;
    double x;

    kernel.resize(k);
    for (int i = 0; i < kernel.size(); i++) {
        x = (double) (i - k / 2);
        kernel[i] = exp(-1 * x * x / (2 * sig * sig)) / sqrt(2 * sig * sig * M_PI);
    }
}


void convolveHeight(thrust::host_vector<pixel>& out, const pixel* sig, double* kernel, short& width, short& height, int kernelSize) {
    short newHeight = height - kernelSize + 1;
    double resultR;
    double resultG;
    double resultB;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < newHeight; j++) {
            resultR = 0.0f;
            resultG = 0.0f;
            resultB = 0.0f;
            for (int k = 0; k < kernelSize; k++) {
                resultR += sig[i + (j + k) * width].r * kernel[k];
                resultG += sig[i + (j + k) * width].g * kernel[k];
                resultB += sig[i + (j + k) * width].b * kernel[k];
            }
            out[i + j * width].r = resultR;
            out[i + j * width].g = resultG;
            out[i + j * width].b = resultB;
        }
    }
    height = newHeight;
}


void convolveWidth(thrust::host_vector<pixel>& out, const pixel* sig, const double* kernel, short& width, short& height, int kernelSize) {
    short newWidth = width - kernelSize + 1;
    double resultR;
    double resultG;
    double resultB;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < newWidth; j++) {
            resultR = 0.0f;
            resultG = 0.0f;
            resultB = 0.0f;
            for (int k = 0; k < kernelSize; k++) {
                resultR += sig[i * width + j + k].r * kernel[k];
                resultG += sig[i * width + j + k].g * kernel[k];
                resultB += sig[i * width + j + k].b * kernel[k];
            }
            out[i * newWidth + j].r = resultR;
            out[i * newWidth + j].g = resultG;
            out[i * newWidth + j].b = resultB;
        }
    }
    width = newWidth;
}


////************************************** GPU implementations ********************************************
__global__
void convolveHeightKernel() {

}

__global__
void convolveWidthKernel(pixel* out, pixel* sig, double* kernel, const short width,
                         const short newWidth, const short height, const int kernelSize) {
    size_t ix = blockDim.x * blockIdx.x + threadIdx.x;
    size_t iy = blockDim.y * blockIdx.y + threadIdx.y;
    double r = 0;
    double g = 0;
    double b = 0;

    if ((ix < newWidth) && (iy < height)) {

        for (size_t k = 0; k < kernelSize; k++) {
            r += sig[iy * width + ix + k].r * kernel[k];
            g += sig[iy * width + ix + k].g * kernel[k];
            b += sig[iy * width + ix + k].b * kernel[k];
        }
        out[iy * newWidth + ix].r = r;
        out[iy * newWidth + ix].g = g;
        out[iy * newWidth + ix].b = b;

    }
}

thrust::host_vector<pixel> convolve2dGpu(const thrust::host_vector<pixel> sig, const thrust::host_vector<double>kernel,
                                         short& width, short& height, int kernelSize){
    short newWidth, newHeight;

    ////********************************** perform convolution along the width ****************************
    //// define device variables for convolution along the width
    newWidth = width - kernelSize + 1;
    thrust::device_vector<pixel> out1Gpu(newWidth * height);
    pixel* out1Ptr = thrust::raw_pointer_cast(out1Gpu.data());

    thrust::device_vector<pixel> in1 = sig;
    pixel* in1Ptr = thrust::raw_pointer_cast(in1.data());

    thrust::device_vector<double> convKernelGpu = kernel;
    double* kernelPtr = thrust::raw_pointer_cast(convKernelGpu.data());

    //// define kernel launch parameters for convolution along the width
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    dim3 threads(devProp.maxThreadsPerBlock/32, 32);
    dim3 blocks(newWidth/threads.x + 1, height/threads.y + 1);

    convolveWidthKernel<<<blocks, threads>>>(out1Ptr, in1Ptr, kernelPtr, width, newWidth, height, kernelSize);
    cudaDeviceSynchronize();

    thrust::host_vector<pixel> outHost = out1Gpu;

    return outHost;
}


int main(int argc, char* argv[]) {
    ////********************************** parse command line arguments **********************************
    if (argc != 3)
        return 1;
    string filename(argv[1]);
    int sigma = atoi(argv[2]);

    ////********************************** declare variables *********************************************
    FileHandler handler;
    short width;
    short height;

    ////********************************** generate gaussian kernel **************************************
    thrust::host_vector<double> kernel;
    getGaussianKernel(kernel, sigma);
    double* kernelPtr = &kernel[0];

    ////********************************** load image ****************************************************
    thrust::host_vector<char> imageRaw = handler.loadImage(filename, width, height);
    const short widthInit = width;
    const short heightInit = height;
    Image image(imageRaw, width, height);
    pixel* imagePtr = image.getPointer();

    //////********************************** perform convolution(CPU) **************************************
    auto start = timer::now();
    thrust::host_vector<pixel> out1Cpu;
    out1Cpu.resize((width - kernel.size() + 1) * height);
    convolveWidth(out1Cpu, imagePtr, kernelPtr, width, height, kernel.size());

    //thrust::host_vector<pixel> out2Cpu;
    //out2Cpu.resize(width * (height - kernel.size() + 1));
    //convolveHeight(out2Cpu, &out1Cpu[0], kernelPtr, width, height, kernel.size());
    auto end = timer::now();

    // report computation time
    std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
    cout << "CPU Elapsed time: " << t.count() << "ms" << endl;

    //////********************************** save data *****************************************************
    //Image output(out2Cpu, width, height);
    //thrust::host_vector<char> bytes;
    //output.toBytes(bytes);
    //handler.saveImage(bytes, "./resultCpu.tga", width, height);

    ////********************************** perform convolution(GPU) **************************************
    width = widthInit;
    height = heightInit;
    thrust::host_vector<pixel> outGpu = convolve2dGpu(image.getImage(), kernel, width, height, kernel.size()) ;
    checkCalcError(out1Cpu, outGpu);

    return 0;
}
