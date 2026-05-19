#include<bits/stdc++.h>
#define int long long
using namespace std;
const int MOD = 998244353;
const int G = 3;

int fpowMOD(int a , int n , int p){
    int res = 1; a %= p;
    while(n){
        if(n & 1)res = res * a % p;
        a = a * a % p;
        n >>= 1;
    }
    return res;
}

void NTT(vector<int>& A , int n , bool inv){
    for(int i = 1 , j = 0;i<n;++i){
        int bit = n >> 1;
        for(;j & bit;bit >>= 1)j ^= bit;
        j ^= bit;
        if(i < j)swap(A[i], A[j]);
    }
    for(int m = 2;m <= n;m <<= 1){
        int wn = inv ? fpowMOD(G , MOD - 1 - (MOD - 1) / m , MOD)
                     : fpowMOD(G , (MOD - 1) / m , MOD);
        for(int i=0;i<n;i += m){
            int wk = 1;
            for (int j=0;j<m / 2;++j){
                int x = A[i + j] , y = A[i + j + m / 2] * wk % MOD;
                A[i + j]         = (x + y) % MOD;
                A[i + j + m / 2] = (x - y + MOD) % MOD;
                wk = wk * wn % MOD;
            }
        }
    }
    if(inv){
        int ninv = fpowMOD(n, MOD - 2, MOD);
        for (auto& x : A) x = x * ninv % MOD;
    }
}

vector<int> PolyMul(vector<int>& a, vector<int>& b){
    int n = 1;
    int sz = a.size() + b.size() - 1;
    while(n<sz)n <<= 1;

    vector<int>A(n) , B(n);
    for (int i=0;i<(int)a.size();++i)A[i] = a[i];
    for (int i=0;i<(int)b.size();++i)B[i] = b[i];

    NTT(A , n , false);
    NTT(B , n , false);

    for(int i=0;i<n;++i)A[i] = A[i] * B[i] % MOD;

    NTT(A , n , true);

    return vector<int>(A.begin(), A.begin() + sz);
}