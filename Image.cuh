//
// Created by ragib1481 on 10/22/22.
//

#ifndef ASSIGNMENT2_IMAGE_CUH
#define ASSIGNMENT2_IMAGE_CUH

#include <vector>

struct pixel{
    float r;
    float g;
    float b;
};

class Image {
    short width;
    short height;
    std::vector<pixel> image;
public:
    Image(std::vector<char>& img, short width, short height);
    Image(std::vector<pixel>& img, short width, short height);
    void toBytes(std::vector<char>& bytes);
    pixel* getPointer();
};


#endif //ASSIGNMENT2_IMAGE_CUH
