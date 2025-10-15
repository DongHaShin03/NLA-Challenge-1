#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues> 
#include <fstream>
#include <sstream>
#include <cctype>
#include <algorithm> 

using namespace Eigen;
using namespace std;
using SpMat = SparseMatrix<double>;
using SpVec = VectorXd;

void printVec(int dim, vector<pair<int, double>> v){
    for(int i = 0; i < dim; i++){
        cout << "(" << v[i].first << ", " << v[i].second << ")" << endl; 
    }
}

int countOutDiag(SpMat &m, int dim, int np, int nn){
    int count = 0; 
    for(int i = 0; i < np; i++){
        for(int j = nn; j < np + nn; j++){
            if(abs(m.coeffRef(i, j)) > 1e-15)   
                count++; 
        }
    }
    return count; 
}

void countAndSort(int &neg, int &pos, int dim, vector<pair<int, double>> &v){
    for(int i = 0; i < dim; i++){
        if(v[i].first != i+1 && v[i].second >= 0) 
            continue; 

        if(v[i].second >= 0){
            pos++; 
            continue; 
        }
        else{
            if(v[i].first == i+1)
                neg++; 
            for(int j = i; j < dim; j++){
                if(v[j].second >= 0){
                    pos++; 
                    pair<int, double> tmp = v[i]; 
                    v[i] = v[j]; 
                    v[j] = tmp; 
                    break; 
                }
            }
        }

    }
}

int main(){
    // Task 1

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
    cout << "Frobenius Norm: " << sqrt(Ag.squaredNorm()) << std::endl;

    // Task 2
    /* 
    Dg è una matrice diagonale a dominanza non stretta, il vettore y = 0 perchè ogni riga contiene nella posizione (i,i) la somma degli altri elementi della riga
    e gli altri elementi della riga sono invertiti di segno per l'equazione Lg = Dg - Ag
    
    Lg è simmetrica e SEMI-definita positiva perchè è un Laplaciano-like : è a dominanza diagonale non stretta per righe, i valori della diagonale 
    sono tutti >0 e quelli non diagonali sono tutti < 0.
    La molteplicità dell’autovalore 0 è il numero di componenti connesse, in questo caso avendo tutto il grafo connesso, la molteplicità è 1 solo.
     */

    VectorXd vg(n1);
    vg = Ag * VectorXd::Ones(n1);
    SpMat Dg(n1, n1);
    for (int i = 0; i < n1; ++i) Dg.coeffRef(i,i) = vg(i);

    SpMat Lg = Dg - Ag;
    VectorXd x = VectorXd::Ones(n1);
    VectorXd y = Lg * x;
    cout << "Norm of y: "<<y.norm() << endl;
    
    // Task 3
    MatrixXd L = MatrixXd(Lg); // Sparse --> Dense

    SelfAdjointEigenSolver<MatrixXd> saeigensolver(L); //Solver used in case of a symmetric Matrix
    if (saeigensolver.info() != Success) abort();

    auto eigenvalues = saeigensolver.eigenvalues();
    auto eigenvectors = saeigensolver.eigenvectors();
    cout << "Min eigenvalue: " << eigenvalues.minCoeff() << endl << "Max Eigenvalue: " << eigenvalues.maxCoeff() << endl;
    
    // Task 4
    // in questo caso gli autovalori sono ordinati in ordine crescente e abbiamo preso l'autovettore dell'autovalore con il secondo valore minore, è coerente con la specifica
    const double lambda = eigenvalues(1);
    VectorXd fielder = eigenvectors.col(1);

    cout << "Smallest strictly positive eigenvalue of Lg: " << lambda << endl;
    cout << "EigenVector associated to the eigenvalue: " << endl << fielder << endl;

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
    Ds.reserve(VectorXd::Ones(ns));
    for (int i = 0; i < ns; ++i) Ds.coeffRef(i,i) = vs(i);

    SpMat Ls = Ds - As; 
    SpMat B = SpMat(Ls.transpose()) - Ls;
    cout << ((B.norm() < 1e-10) ? "is Symmetric" : "its not symmetric") << endl << "nnz(Ls): " << Ls.nonZeros() << endl;

    // Task 7
    // mpirun -n 4 ./eigen1 Ls.mtx VecChallenge2.txt HisChallenge2.txt -e pi -emaxiter 3000 -etol 1e-08
    // Eigenvalue: 6.013370e+01
    // Iteration Count: 2007
    Ls.coeffRef(0, 0) += 0.2; 
    saveMarket(Ls, "./Ls.mtx");

    // Task 8
    // mu = 29.55
    // Iteration Count : 1063

    // Task 9 
    // Compute Eigenvalue = 1.789090
    // Corrispettive EigenVector -> VecChallenge2.mtx
    // N iterations: 113 - 4
    // Prima abbiamo trovato il secondo autovalore più piccolo con etest5
    // Comando: mpirun -n 4 ./etest5 Ls.mtx VecChallenge2.txt HisChallenge2.txt -emaxiter 10000 -etol 1e-10 -ss 2
    // Dopo abbiamo calcolato con etest1 e con l'inverse method l'autovettore dell'autovalore corrispondente grazie allo shift
    // Comando: mpirun -n 4 ./etest1 Ls.mtx VecChallenge2.mtx HisChallenge2.txt -emaxiter 10000 -etol 1e-10 -e ii -shift 1.789070

    // Test 10
    vector<pair<int, double>> eigV; 
    eigV.reserve(ns); 
    ifstream input("./lis-2.1.10/test/VecChallenge2.mtx");
    if (!input) {
        cerr << "Error opening files\n"; 
        return 1; 
    }

    int position; 
    double val; 

    string dummy;
    getline(input, dummy);  
    getline(input, dummy);  
    while (input >> position >> val) {
        eigV.emplace_back(position, val);
    }

    int nn = 0, np = 0; 
    countAndSort(nn, np, ns, eigV); 
    cout << "np: " << np << endl; 
    cout << "nn: " << nn << endl; 

    SpMat P(ns, ns); 
    for(int i = 0; i < ns; i++) 
        P.coeffRef(i, eigV[i].first - 1) = 1; 
    saveMarket(P, "./permutation.mtx"); 
    
    SpMat Aord = P * As * P.transpose(); 
    saveMarket(Aord, "./Aord.mtx"); 
    int count1 = countOutDiag(Aord, ns, np, nn); 
    int count2 = countOutDiag(As, ns, np, nn); 
    cout << "nnz in A_ord: " << count1 << endl; 
    cout << "nnz in A_s: " << count2 << endl; 
    

    return 0;
}