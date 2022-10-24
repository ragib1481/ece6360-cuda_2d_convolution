//
// Created by ragib1481 on 10/5/22.
//

#ifndef ASSIGNMENT2_FILEHANDLER_CUH
#define ASSIGNMENT2_FILEHANDLER_CUH

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

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
    std::vector<char> loadImage(std::string fileName, short& width, short& height);
    void saveImage(const std::vector<char>& bytes, std::string fileName,
                          const short& width, const short& height);
};


#endif //ASSIGNMENT2_FILEHANDLER_CUH
