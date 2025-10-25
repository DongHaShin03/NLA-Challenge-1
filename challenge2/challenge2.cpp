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
        for(int j = np; j < np + nn; j++){
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
    cout << "Frobenius Norm: " << Ag.norm() << endl;

    // Task 2
    /* 
    Dg è una matrice diagonale a dominanza non stretta, il vettore y = 0 perchè ogni riga contiene nella posizione (i,i) la somma degli altri elementi della riga
    e gli altri elementi della riga sono invertiti di segno per l'equazione Lg = Dg - Ag

    La simmetria della matric Lg è diretta conseguenza della simmetria delle matrici Ag e Dg
    Lg è simmetrica e SEMI-definita positiva
    
    "La molteplicità dell’autovalore 0 è il numero di componenti connesse all'interno del grafo" 
    fonte: "A Tutorial on Spectral Clustering", Ulrike von Luxburg
    In questo caso, avendo tutto il grafo connesso, la molteplicità è 1.
     */

    VectorXd vg(n1);
    vg = Ag * VectorXd::Ones(n1);
    SpMat Dg(n1, n1);
    for (int i = 0; i < n1; i++) 
        Dg.coeffRef(i,i) = vg(i);

    SpMat Lg = Dg - Ag;
    saveMarket(Lg, "./Lg.mtx");
    VectorXd x = VectorXd::Ones(n1);
    VectorXd y = Lg * x;
    cout << "Norm of y: "<< y.norm() << endl;
    
    // Task 3
    SelfAdjointEigenSolver<MatrixXd> saeigensolver(Lg);
    if (saeigensolver.info() != Success) abort();

    auto eigenvalues = saeigensolver.eigenvalues();
    auto eigenvectors = saeigensolver.eigenvectors();
    cout << "Min eigenvalue: " << eigenvalues.head(1) << endl << "Max Eigenvalue: " << eigenvalues.tail(1) << endl;

    
    // Task 4
    // in questo caso gli autovalori sono ordinati in ordine crescente e abbiamo preso l'autovettore dell'autovalore con il secondo valore minore, è coerente con la specifica
    cout << "Smallest strictly positive eigenvalue of Lg: " << eigenvalues(1) << endl;
    cout << "Eigenvector associated to the smallets strictly positive eigenvalue: " << endl << eigenvectors.col(1) << endl;

    // Task 5
    SpMat As; 
    loadMarket(As, "./social.mtx");
    cout << "Frobenius norm of As: " << As.norm() << endl;

    // Task 6 
    int ns = As.rows(); 
    VectorXd vs(ns);
    vs = As * VectorXd::Ones(ns);
    SpMat Ds(ns, ns);
    for (int i = 0; i < ns; i++) 
        Ds.coeffRef(i,i) = vs(i);

    SpMat Ls = Ds - As; 
    SpMat B = SpMat(Ls.transpose()) - Ls;
    cout << "Ls symmetric: " << ((B.norm() < 1e-16) ? "true" : "false") << endl << "nnz(Ls): " << Ls.nonZeros() << endl;

    // Task 7
    // mpirun -n 4 ./eigen1 Ls.mtx VecChallenge2.txt HisChallenge2.txt -e pi -emaxiter 3000 -etol 1e-08
    // Eigenvalue: 6.013370e+01
    // Iteration Count: 2007
    Ls.coeffRef(0, 0) += 0.2; 
    saveMarket(Ls, "./Ls.mtx");

    // Task 8
    // mu = 29.55
    // Iteration Count : 1063
    // if mu > 30 ==> the calculated eigenvalue is the smallest one

    // Task 9 
    // Compute Eigenvalue = 1.789070e+00
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
        cerr << "Error opening file\n"; 
        return 1; 
    }

    int position; 
    double val; 

    string boing;
    getline(input, boing);  
    getline(input, boing);  
    while (input >> position >> val){
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
