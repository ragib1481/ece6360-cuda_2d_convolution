//
// Created by ragib1481 on 10/5/22.
//

#include <sstream>
#include "FileHandler.cuh"

thrust::host_vector<char> FileHandler::loadImage(std::string filename, short& width, short& height) {
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
