//
// Created by ragib1481 on 10/5/22.
//

#include "FileHandler.cuh"

std::vector<std::vector<uint8>> FileHandler::loadImage(std::string fileName) {
    std::vector<std::vector<uint8>> image;
    return image;
}

void FileHandler::saveImage(std::vector<std::vector<uint8>>& image, std::string fileName) {
    int height= image.size();
    int width = image[0].size();
    int k = 0;

    char* bytes = new char[width * height * 3];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            bytes[k++] = image[i][j];
            bytes[k++] = 0;
            bytes[k++] = 0;
        }
    }

    std::ofstream outfile;

    outfile.open(fileName, std::ios::binary | std::ios::out);	// open a binary file
    outfile.put(0);	// id length (field 1)
    outfile.put(0);	// color map type (field 2)
    outfile.put(2);	// image_type (field 3)
    outfile.put(0); outfile.put(0);	// color map field entry index (field 4)
    outfile.put(0); outfile.put(0);	// color map length (field 4)
    outfile.put(0);	// color map entry size (field 4)
    outfile.put(0); outfile.put(0);	// x origin (field 5)
    outfile.put(0); outfile.put(0);	// y origin (field 5)
    outfile.write((char*)&width, 2);	// image width (field 5)
    outfile.write((char*)&height, 2);	// image height (field 5)
    outfile.put(24);	// pixel depth (field 5)
    outfile.put(0);	// image descriptor (field 5)
    outfile.write(bytes, width * height * 3);	// write the image data
    outfile.close();	// close the file
}
