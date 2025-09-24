#include <Eigen/Dense>
#include <iostream>
#include <algorithm>          
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

int main(int argc, char* argv[]) {
  
    //TASK 1
    if (argc < 2){
        std::cerr << "Usage: " << argv[0] << " <image_path>\n";
        return 1;
    }

    const char* input_image_path = argv[1];

    int width, height, channels;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data) {
        std::cerr << "Error: Could not load image " << input_image_path << "\n";
    return 1;
    }

    MatrixXd originalImg(height, width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;  // pixel in posizione (i,j)
            originalImg(i, j) = static_cast<double>(image_data[idx]) / 255.0;
        }
    }

    std::cout << "righe: " << height << "\ncolonne: " << width << "\n";

    //TASK 2

    MatrixXd noise = 40.0 * MatrixXd::Random(height, width);

    MatrixXd noiseImg(height, width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;                 
            noiseImg(i, j) = std::clamp((static_cast<double>(image_data[idx]) + noise(i,j)) / 255.0, 0.0, 1.0);
        }
    }

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noise_image(height, width);
  
    noise_image = noiseImg.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(val * 255.0);
    });

    // Save the image using stb_image_write
    const std::string output_image_path1 = "noise_image.png";
    if (stbi_write_png(output_image_path1.c_str(), width, height, 1,
                     noise_image.data(), width) == 0) {
    std::cerr << "Error: Could not save grayscale image" << std::endl;
    return 1;
  }

    //Task 3
    VectorXd v = Map<VectorXd>(originalImg.data(),originalImg.size());
    VectorXd w = Map<VectorXd>(noiseImg.data(),noiseImg.size());
    std::cout << "Dim di v: " << v.size()<< "\nDim di w: " << w.size() << std::endl;
    std::cout << "Norma di v: " << v.norm() << std::endl;
}
