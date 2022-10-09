//
// Created by ragib1481 on 10/5/22.
//

#ifndef ASSIGNMENT2_FILEHANDLER_CUH
#define ASSIGNMENT2_FILEHANDLER_CUH

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

typedef unsigned int uint8;
class FileHandler {
public:
    static std::vector<std::vector<uint8>> loadImage(std::string fileName);
    static void saveImage(std::vector<std::vector<uint8>>& image, std::string fileName);
};


#endif //ASSIGNMENT2_FILEHANDLER_CUH
