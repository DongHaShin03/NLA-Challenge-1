#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>

using namespace std; 
using namespace Eigen; 
using SpMat = Eigen::SparseMatrix<int>; 
int main(){
    // Task 1.a
    int n1 = 9; 
    SpMat Ag(n1, n1); 
    Ag.coeffRef(0, 1) = 1; 
    Ag.coeffRef(0, 3) = 1; 
    Ag.coeffRef(1, 2) = 1; 
    Ag.coeffRef(2, 3) = 1; 
    Ag.coeffRef(2, 4) = 1; 
    Ag.coeffRef(4, 5) = 1; 
    Ag.coeffRef(4, 7) = 1; 
    Ag.coeffRef(4, 8) = 1; 
    Ag.coeffRef(5, 6) = 1; 
    Ag.coeffRef(6, 7) = 1; 
    Ag.coeffRef(6, 8) = 1; 
    Ag.coeffRef(7, 8) = 1; 

    Ag = SpMat(Ag.transpose()) + Ag; 
    saveMarket(Ag, "./Ag.mtx"); 

    // Task 1.b
    SpMat tempA = SpMat(Ag.transpose()) * Ag; 
    double trace; 
    for(int i = 0; i < n1; i++)
        trace += Ag.coeffRef(i, i); 
    
    double frobNorm = sqrt(trace); 
    cout << "Frobenius Norm: " << frobNorm << endl; 

    

}