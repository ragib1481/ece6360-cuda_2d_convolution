//
// Created by ragib1481 on 10/22/22.
//

#ifndef ASSIGNMENT2_IMAGE_CUH
#define ASSIGNMENT2_IMAGE_CUH

#include <vector>
#include <iostream>
#include <thrust/host_vector.h>

struct pixel{
    double r;
    double g;
    double b;
};

class Image {
    short width;
    short height;
    thrust::host_vector<pixel> image;
public:
    Image(thrust::host_vector<char>& img, short width, short height);
    Image(thrust::host_vector<pixel>& img, short width, short height);
    void toBytes(thrust::host_vector<char>& bytes);
    pixel* getPointer();
    thrust::host_vector<pixel> getImage();
};


#endif //ASSIGNMENT2_IMAGE_CUH
