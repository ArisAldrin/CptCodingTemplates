#include <bits/stdc++.h>
#include "Fast_Pow.hpp"
#define int long long
using namespace std;

class Comb{
private:
    int P;
public:
    vector<int> fact , invf;
    Comb(int n , int p = MOD) : P(p) {
        fact.assign(n + 1 , 1); invf.assign(n + 1 , 1);
        for(int i=1;i<=n;++i)fact[i] = fact[i - 1] * i % P;
        invf[n] = inv(fact[n] , P);
        for(int i=n - 1;i>=1;--i)invf[i] = invf[i + 1] * (i + 1) % P;
    }
    int C(int M , int N){ return fact[M] % P * invf[N] % P * invf[M - N] % P; }
    int A(int M , int N){ return fact[M] % P * invf[M - N] % P; }
    int Q(int M , int N){ return fact[M] % P * inv(N , P) % P * invf[M - N] % P; };
    int invC(int M , int N){ return inv(C(M , N) , P) % P; }
    int invA(int M , int N){ return inv(A(M , N) , P) % P; }
    int invQ(int M , int N){ return inv(Q(M , N) , P) % P; }
};

class Comb2{
private:
    int P;
public:
    vector<vector<int> > C;
    Comb2(int n , int m , int p = MOD) : P(p) { // long long overflow when n > 66 without P
        C.assign(n + 1 , vector<int>(m + 1 , 0));
        for(int i=0;i<=n;++i)C[i][0] = 1;
        for(int i=1;i<=n;++i)for(int j=1;j<=min(i , m);++j)C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % P;
    }
};