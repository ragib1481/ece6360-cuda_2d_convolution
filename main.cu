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

thrust::host_vector<pixel> checkCalcError(const thrust::host_vector<pixel> cpuOut, const thrust::host_vector<pixel> gpuOut) {
    if (cpuOut.size() != gpuOut.size()) {
        cout << "Size mismatch" << endl;
        return thrust::host_vector<pixel>();
    }
    thrust::host_vector<pixel> errors(cpuOut.size());
    float error = 0.0;
    for (size_t i = 0; i < cpuOut.size(); i++) {
        errors[i].r = abs(cpuOut[i].r - gpuOut[i].r);
        errors[i].g = abs(cpuOut[i].g - gpuOut[i].g);
        errors[i].b = abs(cpuOut[i].b - gpuOut[i].b);
        //error += (errors[i].r + errors[i].g + errors[i].b);
        error += abs(cpuOut[i].r - gpuOut[i].r);
        error += abs(cpuOut[i].g - gpuOut[i].g);
        error += abs(cpuOut[i].b - gpuOut[i].b);
    }
    cout << "Total absolute error: " << error << endl;

    return errors;
}


void getGaussianFilter(thrust::host_vector<float>& filter, int sigma) {
    /* This function returns a vector containing a 1d gaussian filter assuming 0 mean.
     * Since we are assuming the filter is symmetric, this is a more
     * efficient approach instead of using a 2d gaussian filter.
     */
    int k = 6 * sigma + 1;
    float sig = (float) sigma;
    float x;

    filter.resize(k);
    for (int i = 0; i < filter.size(); i++) {
        x = (float) (i - k / 2);
        filter[i] = (float)(exp(-1.0 * x * x / (2.0 * sig * sig)) / sqrt(2.0 * sig * sig * M_PI));
        //filter[i] = 1.0 / (float)(256);
    }
}


void convolveHeight(thrust::host_vector<pixel>& out, const pixel* sig, const float* filter, short& width, short& height, int filterSize) {
    short newHeight = height - filterSize + 1;
    float resultR;
    float resultG;
    float resultB;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < newHeight; j++) {
            resultR = 0.0;
            resultG = 0.0;
            resultB = 0.0;
            for (int k = 0; k < filterSize; k++) {
                resultR += sig[i + (j + k) * width].r * filter[k];
                resultG += sig[i + (j + k) * width].g * filter[k];
                resultB += sig[i + (j + k) * width].b * filter[k];
            }
            out[i + j * width].r = resultR;
            out[i + j * width].g = resultG;
            out[i + j * width].b = resultB;
        }
    }
    height = newHeight;
}


void convolveWidth(thrust::host_vector<pixel>& out, const pixel* sig, const float* filter, short& width, short& height, int filterSize) {
    short newWidth = width - filterSize + 1;
    float resultR;
    float resultG;
    float resultB;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < newWidth; j++) {
            resultR = 0.0;
            resultG = 0.0;
            resultB = 0.0;
            for (int k = 0; k < filterSize; k++) {
                resultR += sig[i * width + j + k].r * filter[k];
                resultG += sig[i * width + j + k].g * filter[k];
                resultB += sig[i * width + j + k].b * filter[k];
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
void convolveHeightKernel(pixel* out, const pixel* sig, const float* filter, const short width,
                          const short height, const short newHeight, const int filterSize) {
    extern __shared__ unsigned char sharedPtr[];                                // pointer to the shared memory
    float* filterShared = (float*)sharedPtr;                                  // ptr to kernel shared memory
    size_t ix = blockDim.x * blockIdx.x + threadIdx.x;
    size_t iy = blockDim.y * blockIdx.y + threadIdx.y;
    float r = 0.0;
    float g = 0.0;
    float b = 0.0;
    size_t idx;

    if ((ix < width) && (iy < newHeight)) {

        // copy filter to the shared memory
        for (int i = (threadIdx.y * blockDim.x + threadIdx.x); i < filterSize; i += blockDim.x * blockDim.y)
            filterShared[i] = filter[i];
        __syncthreads();

        for (int k = 0; k < filterSize; k++) {
            idx = ix + (iy + k) * width;
            r += sig[idx].r * filter[k];
            g += sig[idx].g * filter[k];
            b += sig[idx].b * filter[k];
        }
        idx = ix + iy * width;
        out[idx].r = r;
        out[idx].g = g;
        out[idx].b = b;
    }
}

__global__
void convolveWidthKernel(pixel* out, const pixel* sig, const float* filter, const short width,
                         const short newWidth, const short height, const int filterSize) {
    extern __shared__ unsigned char sharedPtr[];                                // pointer to the shared memory
    float* filterShared = (float*)sharedPtr;                                  // ptr to kernel shared memory
    size_t ix = blockDim.x * blockIdx.x + threadIdx.x;                          // calculate index along x dimension
    size_t iy = blockDim.y * blockIdx.y + threadIdx.y;                          // calculate index along y dimension
    size_t idx;                                                                 // temporary variable to hold index

    float r = 0.0;
    float g = 0.0;
    float b = 0.0;

    // check if the indices are out scope
    if ((ix < newWidth) && (iy < height)) {

        // copy filter to the shared memory
        for (int i = (threadIdx.y * blockDim.x + threadIdx.x); i < filterSize; i += blockDim.x * blockDim.y)
            filterShared[i] = filter[i];
        __syncthreads();

        // perform convolution
        for (int k = 0; k < filterSize; k++) {
            idx = iy * width + ix + k;
            r += sig[idx].r * filter[k];
            g += sig[idx].g * filter[k];
            b += sig[idx].b * filter[k];
        }

        // put data into the output
        idx = iy * newWidth + ix;
        out[idx].r = r;
        out[idx].g = g;
        out[idx].b = b;
    }
}

thrust::host_vector<pixel> convolve2dGpu(const thrust::host_vector<pixel> sig, const thrust::host_vector<float>filter,
                                         short& width, short& height, const int filterSize, FileHandler& handler){
    short newWidth, newHeight;

    ////********************************** perform convolution along the width ****************************
    //// define device variables for convolution along the width
    newWidth = width - filterSize + 1;
    thrust::device_vector<pixel> out1Gpu(newWidth * height);
    pixel* out1Ptr = thrust::raw_pointer_cast(out1Gpu.data());

    thrust::device_vector<pixel> in1 = sig;
    pixel* in1Ptr = thrust::raw_pointer_cast(in1.data());

    thrust::device_vector<float> convFilterGpu = filter;
    float* filterPtr = thrust::raw_pointer_cast(convFilterGpu.data());

    //// define filter launch parameters for convolution along the width
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    dim3 blockDimension(32, 16, 1);
    dim3 gridDimension((newWidth + blockDimension.x - 1) / blockDimension.x,
                       (height + blockDimension.y - 1) / blockDimension.y, 1);

    // shared memory size
    size_t sharedBytes = filterSize * sizeof(float);

    convolveWidthKernel<<<gridDimension, blockDimension, sharedBytes>>>
                            (out1Ptr, in1Ptr, filterPtr, width, newWidth, height, filterSize);
    cudaDeviceSynchronize();
    width = newWidth;

    ////********************************** perform convolution along the height ***************************
    newHeight = height - filterSize + 1;
    gridDimension.y = (newHeight + blockDimension.y - 1) / blockDimension.y;

    thrust::device_vector<pixel> out2Gpu(newWidth * newHeight);
    pixel* out2Ptr = thrust::raw_pointer_cast(out2Gpu.data());
    convolveHeightKernel<<<gridDimension, blockDimension, sharedBytes>>>
                            (out2Ptr, out1Ptr, filterPtr, width, height, newHeight, filterSize);
    cudaDeviceSynchronize();
    height = newHeight;

    thrust::host_vector<pixel> outHost = out2Gpu;

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

    ////********************************** generate gaussian filter **************************************
    thrust::host_vector<float> filter;
    getGaussianFilter(filter, sigma);
    float* kernelPtr = &filter[0];

    ////********************************** load image ****************************************************
    thrust::host_vector<char> imageRaw = handler.loadImage(filename, width, height);
    const short widthInit = width;
    const short heightInit = height;
    Image image(imageRaw, width, height);
    pixel* imagePtr = image.getPointer();

    //////********************************** perform convolution(CPU) **************************************
    auto start = timer::now();
    thrust::host_vector<pixel> out1Cpu;
    out1Cpu.resize((width - filter.size() + 1) * height);
    convolveWidth(out1Cpu, imagePtr, kernelPtr, width, height, filter.size());

    thrust::host_vector<pixel> out2Cpu;
    out2Cpu.resize(width * (height - filter.size() + 1));
    convolveHeight(out2Cpu, &out1Cpu[0], kernelPtr, width, height, filter.size());
    auto end = timer::now();

    // report computation time
    std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
    cout << "CPU Elapsed time: " << t.count() << "ms" << endl;

    //////********************************** save data (CPU) *********************************************
    Image outputCpu(out2Cpu, width, height);
    thrust::host_vector<char> bytesCpu;
    outputCpu.toBytes(bytesCpu, true);
    handler.saveImage(bytesCpu, "./resultCpu.tga", width, height);

    ////********************************** perform convolution(GPU) **************************************
    width = widthInit;
    height = heightInit;
    start = timer::now();
    thrust::host_vector<pixel> outGpu = convolve2dGpu(image.getImage(), filter, width, height, filter.size(), handler) ;
    end = timer::now();

    t = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
    cout << "GPU Elapsed time: " << t.count() << "ms" << endl;

    //////********************************** save data (GPU) *********************************************
    Image outputGpu(outGpu, width, height);
    thrust::host_vector<char> bytesGpu;
    outputGpu.toBytes(bytesGpu, true);
    handler.saveImage(bytesGpu, "./resultGpu.tga", width, height);

    //////********************************** save data (err) *********************************************
    thrust::host_vector<pixel> errors = checkCalcError(out2Cpu, outGpu);
    return 0;
}
