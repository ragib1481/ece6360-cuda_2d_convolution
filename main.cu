#include <iostream>
#include <string>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "Image.cuh"
#include "FileHandler.cuh"


using namespace std;


void getGaussianKernel(thrust::host_vector<float>& kernel, int sigma) {
    /* This function returns a vector containing a 1d gaussian kernel assuming 0 mean.
     * Since we are assuming the kernel is symmetric, this is a more
     * efficient approach instead of using a 2d gaussian kernel.
     */
    int k = 6 * sigma + 1;
    float sig = (float) sigma;
    float x;

    kernel.resize(k);
    for (int i = 0; i < kernel.size(); i++) {
        x = (float) (i - k / 2);
        kernel[i] = exp(-1 * x * x / (2 * sig * sig)) / sqrt(2 * sig * sig * M_PI);
    }
}


void convolveHeight(thrust::host_vector<pixel>& out, pixel* sig, float* kernel, short& width, short& height, int kernelSize) {
    short newHeight = height - kernelSize + 1;
    float resultR;
    float resultG;
    float resultB;

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


void convolveWidth(thrust::host_vector<pixel>& out, pixel* sig, const float* kernel, short& width, short& height, int kernelSize) {
    short newWidth = width - kernelSize + 1;
    float resultR;
    float resultG;
    float resultB;

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
void convolveWidthKernel(pixel* out, pixel* sig, float* kernel, short width, short newWidth, short height, int kernelSize) {

}

thrust::host_vector<pixel> convolve2dGpu(pixel* sig, const float* kernel, short& width, short& height, int kernelSize){
    short newWidth, newHeight;

    ////********************************** perform convolution along the width ****************************
    //// define device variables for convolution along the width
    newWidth = width - kernelSize + 1;
    thrust::device_vector<pixel> out1(newWidth * height);
    pixel* out1Ptr = thrust::raw_pointer_cast(out1.data());

    //// define kernel launch parameters for convolution along the width
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    cout << devProp.maxThreadsPerBlock << endl;
    cout << devProp.maxGridSize[0] << ", " << devProp.maxGridSize[1] << ", " << devProp.maxGridSize[2] << endl;
    cout << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1] << ", " << devProp.maxThreadsDim[2] << endl;
    return thrust::host_vector<pixel>();
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
    thrust::host_vector<float> kernel;
    getGaussianKernel(kernel, sigma);
    float* kernelPtr = &kernel[0];

    ////********************************** load image ****************************************************
    thrust::host_vector<char> imageRaw = handler.loadImage(filename, width, height);
    const short widthInit = width;
    const short heightInit = height;
    Image image(imageRaw, width, height);
    pixel* imagePtr = image.getPointer();

    //////********************************** perform convolution(CPU) **************************************
    //thrust::host_vector<pixel> out1Cpu;
    //out1Cpu.resize((width - kernel.size() + 1) * height);
    //convolveWidth(out1Cpu, imagePtr, kernelPtr, width, height, kernel.size());

    //thrust::host_vector<pixel> out2Cpu;
    //out2Cpu.resize(width * (height - kernel.size() + 1));
    //convolveHeight(out2Cpu, &out1Cpu[0], kernelPtr, width, height, kernel.size());

    //////********************************** save data *****************************************************
    //Image output(out2Cpu, width, height);
    //thrust::host_vector<char> bytes;
    //output.toBytes(bytes);
    //handler.saveImage(bytes, "./resultCpu.tga", width, height);

    ////********************************** perform convolution(GPU) **************************************
    width = widthInit;
    height = heightInit;
    thrust::host_vector<pixel> out = convolve2dGpu(imagePtr, kernelPtr, width, height, kernel.size()) ;

    return 0;
}
