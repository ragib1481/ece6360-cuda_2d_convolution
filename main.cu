/* Written by: Ragib Ishrak,
 *              Department of ECE,
 *              University of Houston
 *              rishrak@uh.edu
 *
 * This program shows the difference between the computation time of a 2d image on the cpu and the gpu.
 * To get the same result on the cpu and gpu nvcc should be passed the flag "-fmad=false",
 * this will make the computation slower. To make the computation faster on the gpu omit the above flag.
 */

#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


using namespace std;
typedef std::chrono::high_resolution_clock timer;

////************************************** Image utility class ********************************************
// pixel data structure containing r,g,b value
struct pixel{
    float r = 0.0;
    float g = 0.0;
    float b = 0.0;
};

class Image {
    short width;                                            // width of the image
    short height;                                           // height of the image
    thrust::host_vector<pixel> image;                       // 1d vector containing the 2d image in the row major order
public:
    Image(thrust::host_vector<char>& img, short width, short height);
    Image(thrust::host_vector<pixel>& img, short width, short height);
    void toBytes(thrust::host_vector<char>& bytes, bool scale=false);
    pixel* getPointer();
    thrust::host_vector<pixel> getImage();
};

Image::Image(thrust::host_vector<char> &img, short width, short height) {
    /* this function converts the bytes read from the file into an object of the image class.
     * The underlined data structure contains the image as 1d vector in the row major order.
     * The datatype is a pixel structure containing r,g,b values as floats.
     * */
    this->width = width;
    this->height = height;
    image.resize(width * height);
    for (size_t i=0; i < width * height; i++) {
        // scale the 0-255 uint value to floats in the range 0.0-1.0
        image[i].b = ((float)(unsigned char)img[3 * i]) / 255.0;
        image[i].g = ((float)(unsigned char)img[3 * i + 1]) / 255.0;
        image[i].r = ((float)(unsigned char)img[3 * i + 2]) / 255.0;
    }
}

Image::Image(thrust::host_vector<pixel> &img, short width, short height) {
    // copy constructor
    this->image = img;
}

pixel* Image::getPointer() {
    // return the pointer to the vector containing the image
    return &image[0];
}

void Image::toBytes(thrust::host_vector<char>& bytes, bool scale) {
    /* This function converts the vector<pixel> containing the image to a vector of char bytes to be saved as
     * a tga file. If the scale parameter is true then the image is scaled in the range 0.0-1.0 and is converted to
     * char in the range 0-255. otherwise scaling is ignored.
     * */

    bytes.resize(image.size() * 3);

    float max = 0.0;
    // find the maximum pixel r/g/b value for subsequent scaling
    if (scale) {
        for (size_t i = 0; i < image.size(); i++) {
            if (image[i].b > max)
                max = image[i].b;
            if (image[i].g > max)
                max = image[i].g;
            if (image[i].r > max)
                max = image[i].r;
        }
    }

    for (size_t i = 0; i < image.size(); i++){
        // perform scaling and convert to char
        if (scale) {
            bytes[3 * i] = (char) (255 * image[i].b / max);
            bytes[3 * i + 1] = (char) (255 * image[i].g / max);
            bytes[3 * i + 2] = (char) (255 * image[i].r / max);
        }
        // ignore scaling and convert to char
        else{
            bytes[3 * i] = (char) (image[i].b * 255.0);
            bytes[3 * i + 1] = (char) (image[i].g * 255.0);
            bytes[3 * i + 2] = (char) (image[i].r * 255.0);
        }
    }
}

thrust::host_vector<pixel> Image::getImage() {
    // return the vector containing the image
    return image;
}


////************************************** Filehandler utility class **************************************
class FileHandler {
    char id_length;
    char cmap_type;
    char image_type;
    char field_entry_a, field_entry_b;
    char map_length_a, map_length_b;
    char map_size;
    char origin_x_a, origin_x_b;
    char origin_y_a, origin_y_b;
    char pixel_depth;
    char descriptor;

public:
    thrust::host_vector<char> loadImage(std::string fileName, short& width, short& height);
    void saveImage(const thrust::host_vector<char>& bytes, std::string fileName,
                   const short& width, const short& height);
};

thrust::host_vector<char> FileHandler::loadImage(std::string filename, short& width, short& height) {
    /* Load the image from a tga file return as a vector of char data*/
    std::ifstream infile;
    infile.open(filename, std::ios::binary | std::ios::out);        // open the file for binary writing
    if (!infile.is_open()) {
        std::cout << "ERROR: Unable to open file " << filename << std::endl;
        return thrust::host_vector<char>();
    }
    infile.get(id_length);                            // id length (field 1)
    infile.get(cmap_type);                            // color map type (field 2)
    infile.get(image_type);                        // image_type (field 3)
    infile.get(field_entry_a);
    infile.get(field_entry_b);                        // color map field entry index (field 4)

    infile.get(map_length_a);
    infile.get(map_length_b);                        // color map field entry index (field 4)

    infile.get(map_size);                            // color map entry size (field 4)

    infile.get(origin_x_a);
    infile.get(origin_x_b);                        // x origin (field 5)

    infile.get(origin_y_a);
    infile.get(origin_y_b);                        // x origin (field 5)

    infile.read((char*)&width, 2);
    infile.read((char*)&height, 2);

    infile.get(pixel_depth);
    infile.get(descriptor);

    thrust::host_vector<char> bytes(width * height * 3);
    infile.read(&bytes[0], width * height * 3);
    infile.close();                    // close the file

    return bytes;
}

void FileHandler::saveImage(const thrust::host_vector<char>& bytes, std::string fileName, const short& width, const short& height) {
    /* save the vector of char as an uncompressed tga file*/
    std::ofstream outfile;

    outfile.open(fileName, std::ios::binary | std::ios::out);	// open a binary file
    outfile.put(id_length);	// id length (field 1)
    outfile.put(cmap_type);	// color map type (field 2)
    outfile.put(image_type);	// image_type (field 3)
    outfile.put(field_entry_a); outfile.put(field_entry_b);	// color map field entry index (field 4)
    outfile.put(map_length_a); outfile.put(map_length_b);	// color map length (field 4)
    outfile.put(map_size);	// color map entry size (field 4)
    outfile.put(origin_x_a); outfile.put(origin_x_b);	// x origin (field 5)
    outfile.put(origin_y_a); outfile.put(origin_y_b);	// y origin (field 5)
    outfile.write((char*)&width, 2);	// image width (field 5)
    outfile.write((char*)&height, 2);	// image height (field 5)
    outfile.put(pixel_depth);	// pixel depth (field 5)
    outfile.put(descriptor);	// image descriptor (field 5)
    outfile.write(&bytes[0], width * height * 3);	// write the image data
    outfile.close();	// close the file
}


////************************************** helper functions ***********************************************
thrust::host_vector<pixel> checkCalcError(const thrust::host_vector<pixel> cpuOut, const thrust::host_vector<pixel> gpuOut) {
    /* This function calculates the error of calculation between cpu computation and gpu computation.
     * The absolution value of the error for each pixel is summed and reported.
     */

    // check if the size of the convolved images match
    if (cpuOut.size() != gpuOut.size()) {
        cout << "Size mismatch" << endl;
        return thrust::host_vector<pixel>();
    }
    thrust::host_vector<pixel> errors(cpuOut.size());

    float error = 0.0;                                      // variable to accumulate the error
    for (size_t i = 0; i < cpuOut.size(); i++) {
        error += abs(cpuOut[i].r - gpuOut[i].r);        // calculate for the red value
        error += abs(cpuOut[i].g - gpuOut[i].g);        // calculate for the green value
        error += abs(cpuOut[i].b - gpuOut[i].b);        // calculate for the blue value
    }
    cout << "Total absolute error: " << error << endl;     // report the error
    return errors;
}


////************************************** function for generating gaussian filter *************************
void getGaussianFilter(thrust::host_vector<float>& filter, int sigma) {
    /* This function returns a vector containing a 1d gaussian filter assuming 0 mean.
     * Since we are assuming the filter is symmetric, this is a more
     * efficient approach instead of using a 2d gaussian filter.
     */
    int k = 6 * sigma + 1;                              // calculate the filter size from the variance
    float sig = (float) sigma;                          // convert to float
    float x;                                            // temporary variable

    filter.resize(k);
    for (int i = 0; i < filter.size(); i++) {
        x = (float) (i - k / 2);                // calculate location on the number line considering the 0 at the center.
        // calculate gaussian
        filter[i] = (float)(exp(-1.0 * x * x / (2.0 * sig * sig)) / sqrt(2.0 * sig * sig * M_PI));
    }
}


////************************************** CPU implementation of convolution *******************************
void convolveHeight(thrust::host_vector<pixel>& out, const pixel* sig, const float* filter, short& width, short& height, int filterSize) {
    /* Convolve the image along the height on the cpu.
     * The convolution performed is a valid convolution. That is only the part of the convolution where the image is
     * defined calculated. As a result the image returned has smaller height than the original image.
     */
    short newHeight = height - filterSize + 1;              // calculate the height of the image for valid convolution

    // temporary variables for accumulating the convolution results
    float resultR;
    float resultG;
    float resultB;

    size_t idx;                                             // idx variable to hold the 1d index of the present pixel

    /* Outer loop strides along the width and the inner loop strides along the height
     */
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < newHeight; j++) {
            // at the start of convolution for each output pixel set the accumulators to zero
            resultR = 0.0;
            resultG = 0.0;
            resultB = 0.0;

            // loop to perform the convolution
            for (int k = 0; k < filterSize; k++) {
                idx = i + (j + k) * width;              // convert 2d pixel index into 1d vector index for the input image

                // multiply the filter with the pixel and accumulate
                resultR += sig[idx].r * filter[k];
                resultG += sig[idx].g * filter[k];
                resultB += sig[idx].b * filter[k];
            }
            idx = i + j * width;                       // convert 2d pixel index into 1d vector index for the output image

            // put the result in the output image
            out[idx].r = resultR;
            out[idx].g = resultG;
            out[idx].b = resultB;
        }
    }
    height = newHeight;                                 // set the height of the image to the new height
}

void convolveWidth(thrust::host_vector<pixel>& out, const pixel* sig, const float* filter, short& width, short& height, int filterSize) {
    /* Convolve the image along the width on the cpu.
     * The convolution performed is a valid convolution. That is only the part of the convolution where the image is
     * defined calculated. As a result the image returned has smaller width than the original image.
     */
    short newWidth = width - filterSize + 1;                        // width of the image after the convolution

    // temporary variables for accumulating the convolution results
    float resultR;
    float resultG;
    float resultB;

    size_t idx;                                             // idx variable to hold the 1d index of the present pixel

    /* Outer loop strides along the height and the inner loop strides along the width
     */
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < newWidth; j++) {
            resultR = 0.0;
            resultG = 0.0;
            resultB = 0.0;
            for (int k = 0; k < filterSize; k++) {
                idx = i * width + j + k;                    // convert 2d pixel index to 1d vector index in the input image

                // multiply the filter with the pixel and accumulate
                resultR += sig[idx].r * filter[k];
                resultG += sig[idx].g * filter[k];
                resultB += sig[idx].b * filter[k];
            }
            idx = i * newWidth + j;                         // convert 2d pixel index to 1d vector index in the input image

            // put the result in the output image
            out[idx].r = resultR;
            out[idx].g = resultG;
            out[idx].b = resultB;
        }
    }
    width = newWidth;                                       // width of the image after the convolution
}


////************************************** GPU implementation of convolution ******************************
/* declare a filter array in the constant memory of full size since the constant memory cannot be dynamically
 * allocated. If the size of the filter is K then only first K elements of the constant memory array will be used.
 */
__constant__ float filterConstMem[65536 / sizeof(float)];

__global__
void convolveHeightKernel(pixel* out, const pixel* sig, const short width,
                          const short height, const short newHeight, const int filterSize) {
    /* The gpu kernel to calculate the convolution along the height of the image. Each thread of the
     * gpu calculates the convolution result for one pixel of the output image.
     * First the input signal is copied to the shared memory.
     * The filter copied to the constant memory is used to perform the convolution
     */

    extern __shared__ pixel signalSharedMem[];                      // pointer to the shared memory to hold the signal
    size_t ix = blockDim.x * blockIdx.x + threadIdx.x;              // x index of the present pixel in the output image
    size_t iy = blockDim.y * blockIdx.y + threadIdx.y;              // y index of the present pixel in the output image
    size_t idx;                                                     // temp variable to hold the 1d index of the image

    //number of elements to be copied from the global memory to the shared memory along the height for each block
    int numElementToCopy = blockDim.y + filterSize - 1;

    // accumulator variables
    float r = 0.0;
    float g = 0.0;
    float b = 0.0;

    /* Copy signal to the shared memory. This copy is performed for each independent blocks.
     * All threads along the height of each block copies all the signal points required to perform
     * the convolution along the present column of the block.
     */
    for (int i = threadIdx.y; (i < numElementToCopy) && ((blockDim.y * blockIdx.y + i) < height); i += blockDim.y) {
        signalSharedMem[i * blockDim.x + threadIdx.x] = sig[ix + (blockDim.y * blockIdx.y + i) * width];
    }
    __syncthreads();                                            // wait for the all the threads to copy the data

    // since it is possible that more threads are getting launched than necessary check if the index is outof range
    if ((ix < width) && (iy < newHeight)) {

        // loop to calculate the convolution for each pixel
        for (int k = 0; k < filterSize; k++) {
            idx = (threadIdx.y + k) * blockDim.x + threadIdx.x;     // 1d index of the present pixel in the input

            // multiply each pixel with the filter and accumulate the result
            r += signalSharedMem[idx].r * filterConstMem[k];
            g += signalSharedMem[idx].g * filterConstMem[k];
            b += signalSharedMem[idx].b * filterConstMem[k];
        }
        idx = ix + iy * width;                                      // 1d index of the present pixel in the output
        out[idx].r = r;
        out[idx].g = g;
        out[idx].b = b;
    }
}

__global__
void convolveWidthKernel(pixel* out, const pixel* sig, const short width,
                         const short newWidth, const short height, const int filterSize) {
    /* The gpu kernel to calculate the convolution along the width of the image. Each thread of the
     * gpu calculates the convolution result for one pixel of the output image.
     * First the input signal is copied to the shared memory.
     * The filter copied to the constant memory is used to perform the convolution
     */
    extern __shared__ pixel signalSharedMem[];                  // pointer to the shared memory to hold the signal
    size_t ix = blockDim.x * blockIdx.x + threadIdx.x;          // x index of the present pixel in the output image
    size_t iy = blockDim.y * blockIdx.y + threadIdx.y;          // y index of the present pixel in the output image
    size_t idx;                                                 // temp variable to hold the 1d index of the image


    //number of elements to be copied from the global memory to the shared memory along the width for each block
    int numElementToCopy = blockDim.x + filterSize - 1;

    float r = 0.0;
    float g = 0.0;
    float b = 0.0;

    /* Copy signal to the shared memory. This copy is performed for each independent blocks.
     * All threads along the width of each block copies all the signal points required to perform
     * the convolution along the present row of the block.
     */
    for (int i = threadIdx.x; (i < numElementToCopy) && ((blockDim.x * blockIdx.x + i) < width); i += blockDim.x) {
        signalSharedMem[threadIdx.y * numElementToCopy + i] = sig[iy * width + blockIdx.x * blockDim.x + i];
    }
    __syncthreads();                                            // wait for all threads to finish copying the data

    // check if the indices are out scope
    if ((ix < newWidth) && (iy < height)) {

        // perform convolution
        for (int k = 0; k < filterSize; k++) {
            idx = threadIdx.y * numElementToCopy + threadIdx.x + k;     // 1d index of the present pixel in the input image

            // multiply each pixel with the filter and accumulate the result
            r += signalSharedMem[idx].r * filterConstMem[k];
            g += signalSharedMem[idx].g * filterConstMem[k];
            b += signalSharedMem[idx].b * filterConstMem[k];
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

    size_t sharedBytes = blockDimension.y * (blockDimension.x + filterSize - 1) * sizeof(pixel);

    // shared memory size
    if (sharedBytes > devProp.sharedMemPerBlock) {
        cout << "Invalid filter size." << endl;
        thrust::host_vector<pixel> out((width - filterSize + 1) * (height - filterSize + 1));
        for (size_t i = 0; i < out.size(); i++){
            out[i].r = 0.0; out[i].g = 0.0; out[i].b = 0.0;
        }
        return out;
    }

    // copy filter to constant memory
    cudaMemcpyToSymbol(filterConstMem, filterPtr, filterSize * sizeof(float), 0, cudaMemcpyHostToDevice);

    convolveWidthKernel<<<gridDimension, blockDimension, sharedBytes>>>
                            (out1Ptr, in1Ptr, width, newWidth, height, filterSize);
    width = newWidth;

    ////********************************** perform convolution along the height ***************************
    //sharedBytes = filterSize * sizeof(float);
    blockDimension.y = 8;
    sharedBytes = blockDimension.x * (blockDimension.y + filterSize - 1) * sizeof(pixel);
    if (sharedBytes > devProp.sharedMemPerBlock) {
        cout << "Invalid filter size." << endl;
        thrust::host_vector<pixel> out((width - filterSize + 1) * (height - filterSize + 1));
        for (size_t i = 0; i < out.size(); i++){
            out[i].r = 0.0; out[i].g = 0.0; out[i].b = 0.0;
        }
        return out;
    }
    newHeight = height - filterSize + 1;
    gridDimension.y = (newHeight + blockDimension.y - 1) / blockDimension.y;

    thrust::device_vector<pixel> out2Gpu(newWidth * newHeight);
    pixel* out2Ptr = thrust::raw_pointer_cast(out2Gpu.data());
    convolveHeightKernel<<<gridDimension, blockDimension, sharedBytes>>>
                            (out2Ptr, out1Ptr, width, height, newHeight, filterSize);
    height = newHeight;

    thrust::host_vector<pixel> outHost = out2Gpu;

    return outHost;
}


////************************************** main function *************************************************
int main(int argc, char* argv[]) {
    ////********************************** parse command line arguments **********************************
    if (argc != 3)
        return 1;
    string filename(argv[1]);                   // the first argument is the name of the tga image file
    int sigma = atoi(argv[2]);                // the second argument is the value of sigma for the gaussian filter

    ////********************************** declare variables *********************************************
    FileHandler handler;
    short width;
    short height;

    ////********************************** generate gaussian filter **************************************
    thrust::host_vector<float> filter;                  // vector to containing the filter
    getGaussianFilter(filter, sigma);                // generate gaussian filter
    float* kernelPtr = &filter[0];                      // get pointer to the vector containing the filter

    ////********************************** load image ****************************************************
    thrust::host_vector<char> imageRaw = handler.loadImage(filename, width, height);
    // store the initial width and height for later use
    const short widthInit = width;
    const short heightInit = height;

    // convert char bytes to Image class for convenient handling of operations
    Image image(imageRaw, width, height);
    pixel* imagePtr = image.getPointer();

    //////********************************** perform convolution(CPU) **************************************
    auto start = timer::now();              // for timing information

    // vector to hold the result of the image convolution along the width
    thrust::host_vector<pixel> out1Cpu;
    out1Cpu.resize((width - filter.size() + 1) * height);

    // perform the convolution along the width
    convolveWidth(out1Cpu, imagePtr, kernelPtr, width, height, filter.size());

    // vector to hold the result of the image convolution along the height
    thrust::host_vector<pixel> out2Cpu;
    out2Cpu.resize(width * (height - filter.size() + 1));

    // perform the convolution along the height
    convolveHeight(out2Cpu, &out1Cpu[0], kernelPtr, width, height, filter.size());

    auto end = timer::now();                // for timing information

    // report cpu computation time
    std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
    cout << "CPU Elapsed time: " << t.count() << "ms" << endl;

    //////********************************** save data (CPU) *********************************************
    Image outputCpu(out2Cpu, width, height);
    thrust::host_vector<char> bytesCpu;
    outputCpu.toBytes(bytesCpu, true);
    handler.saveImage(bytesCpu, "./resultCpu.tga", width, height);

    ////********************************** perform convolution(GPU) **************************************
    // re-initialize the height and width of the input image
    width = widthInit;
    height = heightInit;

    start = timer::now();                   // for timing information

    // perform the convolution along the height
    thrust::host_vector<pixel> outGpu = convolve2dGpu(image.getImage(), filter, width, height, filter.size(), handler) ;

    end = timer::now();                     // for timing information

    t = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
    cout << "GPU Elapsed time: " << t.count() << "ms" << endl;

    //////********************************** save data (GPU) *********************************************
    Image outputGpu(outGpu, width, height);
    thrust::host_vector<char> bytesGpu;
    outputGpu.toBytes(bytesGpu, true);
    handler.saveImage(bytesGpu, "./resultGpu.tga", width, height);

    //////********************************** calculate error *********************************************
    checkCalcError(out2Cpu, outGpu);
    return 0;
}
