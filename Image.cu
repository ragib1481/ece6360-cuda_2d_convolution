//
// Created by ragib1481 on 10/22/22.
//

#include "Image.cuh"

Image::Image(thrust::host_vector<char> &img, short width, short height) {

    this->width = width;
    this->height = height;
    image.resize(width * height);
    for (size_t i=0; i < width * height; i++) {
        image[i].b = ((float)(unsigned char)img[3 * i]) / 255.0;
        image[i].g = ((float)(unsigned char)img[3 * i + 1]) / 255.0;
        image[i].r = ((float)(unsigned char)img[3 * i + 2]) / 255.0;
    }
}

Image::Image(thrust::host_vector<pixel> &img, short width, short height) {
    this->image = img;
}

pixel* Image::getPointer() {
    return &image[0];
}

void Image::toBytes(thrust::host_vector<char>& bytes, bool scale) {
    bytes.resize(image.size() * 3);

    float max = 0.0;
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
        if (scale) {
            bytes[3 * i] = (char) (255 * image[i].b / max);
            bytes[3 * i + 1] = (char) (255 * image[i].g / max);
            bytes[3 * i + 2] = (char) (255 * image[i].r / max);
        }
        else{
            bytes[3 * i] = (char) (image[i].b * 255.0);
            bytes[3 * i + 1] = (char) (image[i].g * 255.0);
            bytes[3 * i + 2] = (char) (image[i].r * 255.0);
        }
    }
}

thrust::host_vector<pixel> Image::getImage() {
    return image;
}
