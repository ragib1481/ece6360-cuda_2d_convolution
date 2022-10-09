#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <thrust/host_vector.h>


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


void zeroPad(thrust::host_vector<float>& signal, int n) {
    thrust::host_vector<float> zeros(n, 0);
    signal.insert(signal.end(), zeros.begin(), zeros.end());
    zeros.insert(zeros.end(), signal.begin(), signal.end());
    signal = zeros;
}


void convolve1d(thrust::host_vector<float>& out, thrust::host_vector<float>& signal, thrust::host_vector<float>& kernel) {
    float result = 0;
    int n = signal.size() + kernel.size() - 1;

    /* Perform zero padding on the signal. The zero padding is performed assuming the kernel size is
     * smaller than the signal size. */
    zeroPad(signal, kernel.size()-1);

    out.resize(n, 0);

    for (int i = 0; i < out.size(); i++) {
        result = 0;
        for (int j = 0; j < kernel.size(); j++) {
            result += signal[i + j] * kernel[j];
        }
        out[i] = result;
    }
}


int main(int argc, char* argv[]) {
    //********************************** parse command line arguments **********************************
    if (argc != 3)
        return 1;
    string filename(argv[1]);
    int sigma = atoi(argv[2]);

    //********************************** declare variables *********************************************
    thrust::host_vector<float> kernel;
    thrust::host_vector<float> signal;
    thrust::host_vector<float> out;

    //********************************** generate gaussian kernel **************************************
    getGaussianKernel(kernel, sigma);

    //********************************** load image ****************************************************

    //********************************** perform convolution *******************************************
    signal = kernel;
    convolve1d(out, signal, kernel);

    //********************************** save data *****************************************************
    std::ofstream myFile;
    myFile.open("./kernel.csv");
    myFile << "val" << endl;
    for (auto x: kernel)
        myFile << x << endl;
    myFile.close();

    myFile.open("./signal.csv");
    myFile << "val" << endl;
    for (auto x: signal)
        myFile << x << endl;
    myFile.close();

    myFile.open("./out.csv");
    myFile << "val" << endl;
    for (auto x: out)
        myFile << x << endl;
    myFile.close();
    return 0;
}
