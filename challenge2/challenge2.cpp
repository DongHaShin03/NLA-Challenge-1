#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues> 

using namespace std;
using namespace Eigen;
using SpMat = SparseMatrix<double>;   // usa double ovunque

int main() {
    //Task 1
    int n1 = 9;
    SpMat Ag(n1, n1);

    Ag.coeffRef(0,1) = 1.0;
    Ag.coeffRef(0,3) = 1.0;
    Ag.coeffRef(1,2) = 1.0;
    Ag.coeffRef(2,3) = 1.0;
    Ag.coeffRef(2,4) = 1.0;
    Ag.coeffRef(4,5) = 1.0;
    Ag.coeffRef(4,7) = 1.0;
    Ag.coeffRef(4,8) = 1.0;
    Ag.coeffRef(5,6) = 1.0;
    Ag.coeffRef(6,7) = 1.0;
    Ag.coeffRef(6,8) = 1.0;
    Ag.coeffRef(7,8) = 1.0;

    Ag = SpMat(Ag.transpose()) + Ag;
    saveMarket(Ag, "./Ag.mtx");

    double frobNorm = std::sqrt(Ag.squaredNorm());
    std::cout << "Frobenius Norm: " << frobNorm << std::endl;

    //Task 2
    VectorXd vg(n1);

    vg = Ag * VectorXd::Ones(n1);
    saveMarket(vg, "./vg.mtx");

    SpMat Dg(n1, n1);
    Dg.reserve(VectorXi::Ones(n1));
    for (int i = 0; i < n1; ++i) Dg.insert(i,i) = vg(i);

    SpMat Lg = Dg - Ag;
    // Dg è una matrice diagonale a dominanza non stretta, il vettore y = 0 perchè ogni riga contiene nella posizione (i,i) la somma degli altri elementi della riga
    // e gli altri elementi della riga sono invertiti di segno per l'equazione Lg = Dg - Ag

    VectorXd x = VectorXd::Ones(n1);
    VectorXd y = Lg * x;
    cout << y.norm() << endl;
    // Lg è simmetrica e SEMI-definita positiva perchè è un Laplaciano-like : è a dominanza diagonale non stretta per righe, i valori della diagonale sono tutti >0 e quelli non diagonali
    // sono tutti < 0.
    // La molteplicità dell’autovalore 0 è il numero di componenti connesse, in questo caso avendo tutto il grafo connesso, la molteplicità è 1 solo.

    //Task 3
    MatrixXd L = MatrixXd(Lg);

    SelfAdjointEigenSolver<MatrixXd> saeigensolver(L);
    if (saeigensolver.info() != Success) abort();

    auto eigenvalues = saeigensolver.eigenvalues();
    auto eigenvectors = saeigensolver.eigenvectors();
    cout << "Min eigenvalue: " << eigenvalues.minCoeff() << endl << "Max Eigenvalue: " << eigenvalues.maxCoeff() << endl;

    //Task 4
    // in questo caso gli autovalori sono ordinati in ordine crescente e abbiamo preso l'autovettore dell'autovalore con il secondo valore minore, è coerente con la specifica
    VectorXd fielder = eigenvectors.col(1);
    cout << fielder << endl;


}
