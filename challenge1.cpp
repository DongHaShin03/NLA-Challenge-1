#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <algorithm>  
#include <cstdlib>
#include <vector>
#include <cmath>       
#include <string>
#include <unsupported/Eigen/SparseExtra>



#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

SparseMatrix<double> ConvolutionalMatrix(const MatrixXd& H, int dimM, int dimN);
bool isSymmetric(const SparseMatrix<double>& M);

int main(int argc, char* argv[]) {

    // Task 1
    if (argc < 2) {
        std::cerr << "Use: " << argv[0] << " <image_path>\n";
        return 1;
    }

    const char* input_image_path = argv[1];

    int width = 0, height = 0, channels = 0;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
    if (!image_data) {
        std::cerr << "Errore: impossibile caricare l'immagine " << input_image_path << "\n";
        return 1;
    }

    Matrix<double, Dynamic, Dynamic, RowMajor> originalImg(height, width);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            originalImg(i, j) = static_cast<double>(image_data[idx]) / 255.0;
        }
    }

    std::cout << "righe: " << height << "\ncolonne: " << width << "\n";

    // Task 2
    MatrixXd noise = 40.0 * MatrixXd::Random(height, width);
    Matrix<double, Dynamic, Dynamic, RowMajor> noiseImg(height, width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j;
            double with_noise = (static_cast<double>(image_data[idx]) + noise(i, j)) / 255.0;
            noiseImg(i, j) = std::clamp(with_noise, 0.0, 1.0);
        }
    }

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noise_image_u8(height, width);
    noise_image_u8 = noiseImg.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(std::round(std::clamp(val, 0.0, 1.0) * 255.0));
    });

    const std::string output_image_path1 = "noise_image.png";
    if (stbi_write_png(output_image_path1.c_str(), width, height, 1, noise_image_u8.data(), width) == 0) {
        std::cerr << "Errore: impossibile salvare noise_image.png\n";
        stbi_image_free(image_data);
        return 1;
    }

    // Task 3
    VectorXd v = Map<VectorXd>(originalImg.data(), originalImg.size()); 
    VectorXd w = Map<VectorXd>(noiseImg.data(),    noiseImg.size()); 
    std::cout << "Dim di v: " << v.size() << "\nDim di w: " << w.size() << std::endl;
    std::cout << "Norma di v: " << v.norm() << std::endl;

    // Task 4
    MatrixXd Hav1(3, 3);
    Hav1 << 1, 1, 0,
            1, 2, 1,
            0, 1, 1;
    Hav1 = (1.0 / 8.0) * Hav1;

    SparseMatrix<double> A1 = ConvolutionalMatrix(Hav1, height, width);
    std::cout << "A1: " << A1.rows() << "x" << A1.cols()
              << ", nnz = " << A1.nonZeros() << std::endl;

    //Task 5    
    VectorXd y1 = A1 * w;

    
    Matrix<double, Dynamic, Dynamic, RowMajor> filteredD =
        Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(y1.data(), height, width);

    
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> filtered_u8(height, width);
    filtered_u8 = filteredD.unaryExpr([](double val) {
        val = std::clamp(val, 0.0, 1.0);
        return static_cast<unsigned char>(std::round(val * 255.0));
    });

    if (stbi_write_png("filtered.png", width, height, 1, filtered_u8.data(), width) == 0) {
        std::cerr << "Errore: impossibile salvare filtered.png\n";
        stbi_image_free(image_data);
        return 1;
    }

    //Task 6
    MatrixXd Hsh1(3, 3);
    Hsh1 << 0, -2, 0,
            -2, 9, -2,
            0, -2, 0;

    SparseMatrix<double> A2 = ConvolutionalMatrix(Hsh1, height, width);
    std::cout << "A2: " << A2.rows() << "x" << A2.cols()
              << ", nnz = " << A2.nonZeros() << ", simmetrica = " << std::boolalpha << isSymmetric(A2) << std::endl;


    //Task 7
    VectorXd y2 = A2 * v;
    Matrix<double, Dynamic, Dynamic, RowMajor> filteredD_2 =
        Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(y2.data(), height, width);

    
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> filtered_u8_2(height, width);
    filtered_u8_2 = filteredD_2.unaryExpr([](double val) {
        val = std::clamp(val, 0.0, 1.0);
        return static_cast<unsigned char>(std::round(val * 255.0));
    });

    if (stbi_write_png("filtered2.png", width, height, 1, filtered_u8_2.data(), width) == 0) {
        std::cerr << "Errore: impossibile salvare filtered_2.png\n";
        stbi_image_free(image_data);
        return 1;
    }

    //Task 8
    std::string matrixFileOut("./A2.mtx");
    Eigen::saveMarket(A2, matrixFileOut);
    
    FILE* out_w = fopen("w.mtx", "w"); 
    fprintf(out_w, "w: \n"); 
    for(int i = 0; i < A2.rows(); i++) 
        fprintf(out_w, "%d %f\n", i, w(i));
    
    fclose(out_w);


    return 0;
}


SparseMatrix<double> ConvolutionalMatrix(const MatrixXd& H, int dimM, int dimN) {
    const int dimZ = dimM * dimN;
    std::vector<Triplet<double>> triplets;
    triplets.reserve(static_cast<size_t>(dimZ) * static_cast<size_t>(H.size()));

    const int kh = static_cast<int>(H.rows());
    const int kw = static_cast<int>(H.cols());
    const int offi = kh / 2;
    const int offj = kw / 2;

    for (int z = 0; z < dimZ; ++z) {
        int i = z / dimN;
        int j = z % dimN;

        for (int k = 0; k < kh; ++k) {
            for (int l = 0; l < kw; ++l) {
                int i_new = i + k - offi;
                int j_new = j + l - offj;

                if (i_new < 0 || i_new >= dimM || j_new < 0 || j_new >= dimN)
                    continue;

                int z_new = i_new * dimN + j_new;

                double h = H(k,l);
                if (h != 0.0) {
                    triplets.emplace_back(z, z_new, h);
                }
            }
        }
    }

    SparseMatrix<double> A(dimZ, dimZ);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

bool isSymmetric(const SparseMatrix<double>& M) {
    return M.isApprox(M.transpose(), 1e-10);
}
