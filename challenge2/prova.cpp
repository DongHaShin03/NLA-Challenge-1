#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues> 
 
using namespace std;
using namespace Eigen;
using SpMat = SparseMatrix<double>;  
using SpVec = VectorXd; 

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
    vg = Ag * SpVec::Ones(n1);

    SpMat Dg(n1, n1);
    for (int i = 0; i < n1; ++i) Dg.coeffRef(i,i) = vg(i);

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
    SelfAdjointEigenSolver<MatrixXd> eigensolver(Lg);
    if (eigensolver.info() != Success) abort();

    auto eigenvalues = eigensolver.eigenvalues();
    auto eigenvectors = eigensolver.eigenvectors();
    cout << "Min eigenvalue: " << eigenvalues.minCoeff() << endl << "Max Eigenvalue: " << eigenvalues.maxCoeff() << endl;

    //Task 4
    // in questo caso gli autovalori sono ordinati in ordine crescente e abbiamo preso l'autovettore dell'autovalore con il secondo valore minore, è coerente con la specifica
    VectorXd fielder = eigenvectors.col(1);
    cout << fielder << endl;

    // Task 5
    SpMat As; 
    loadMarket(As, "./social.mtx");
    double frobNormS = std::sqrt(As.squaredNorm());
    cout << "Frobenius Norm: " << frobNormS << endl;

    // Task 6
    int ns = As.rows(); 
    VectorXd vs(ns);

    vs = As * VectorXd::Ones(ns);
    SpMat Ds(ns, ns);
    for (int i = 0; i < ns; ++i) Ds.coeffRef(i,i) = vs(i);

    SpMat Ls = Ds - As; 
    SpMat B = SpMat(Ls.transpose()) - Ls;
    std::cout << ((B.norm() < 1e-10) ? "is Symmetric" : "its not symmetric") << endl << "nnz(Ls): " << Ls.nonZeros() << endl;

    // Task 7
    Ls.coeffRef(0, 0) += 0.2; 
    saveMarket(Ls, "./Ls.mtx"); 

    // Task 8 
    // shift 30 è il massimo affinchè l'autovalore massimo calcolato sia lo stesso, però aumenta il numero di iterazioni 
    // shift 29.55 ==> numero minimo di iterazioni (1063 iterazioni per una tolleranza di 1e-8)

    // Task 9
    

}
