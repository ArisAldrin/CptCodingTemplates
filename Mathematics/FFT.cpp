#include<bits/stdc++.h>
#define int long long
using namespace std;
using db = double;
const db PI = acos(-1);

void butterfly(vector<complex<db> >& A , int n){
    vector<int>dp(n, 0);
    for(int i=0;i<n;++i)dp[i] = dp[i / 2] / 2 + ((i & 1) ? n / 2 : 0);
    for(int i=0;i<n;++i)if(i < dp[i])swap(A[i] , A[dp[i]]);
}

void FFT(vector<complex<db> >& A , int n){
    butterfly(A , n);
    for(int m=2;m<=n;m <<= 1){
        complex<db> spin{cos(2 * PI / m) , sin(2 * PI / m)};
        for(int i=0;i<n;i += m){
            complex<db> wk = {1 , 0};
            for(int j=0;j<m / 2;++j){
                complex<db> x = A[i + j] , y = A[i + j + m / 2] * wk;
                A[i + j] = x + y;
                A[i + j + m / 2] = x - y;
                wk *= spin;
            }
        }
    }
}


void IFFT(vector<complex<db> >& A , int n){
    butterfly(A , n);
    for(int m=2;m<=n;m <<= 1){
        complex<db> spin{cos(2 * PI / m) , -sin(2 * PI / m)};
        for(int i=0;i<n;i += m){
            complex<db> wk = {1 , 0};
            for(int j=0;j<m / 2;++j){
                complex<db> x = A[i + j] , y = A[i + j + m / 2] * wk;
                A[i + j] = x + y;
                A[i + j + m / 2] = x - y;
                wk *= spin;
            }
        }
    }
    for(int i=0;i<n;++i)A[i] /= n;
}

vector<int> PolyMul(vector<int>& a, vector<int>& b) {
    int n = 1;
    int sz = a.size() + b.size() - 1;
    while(n < sz) n <<= 1;

    vector<complex<db> >A(n) , B(n);
    for(int i=0;i<(int)a.size();++i)A[i] = a[i];
    for(int i=0;i<(int)b.size();++i)B[i] = b[i];

    FFT(A , n);
    FFT(B , n);

    for(int i=0;i<n;i++)A[i] *= B[i];

    IFFT(A , n);

    vector<int> res(sz);
    for(int i=0;i<sz;++i)res[i] = llround(A[i].real());
    return res;
}

signed main(){
    vector<int>a = {1 , 2 , 3};    // 1 + 2x + 3x^2
    vector<int>b = {1 , 2 , 3};    // 1 + 2x + 3x^2
    auto c = PolyMul(a , b);
    for(auto x:c)cout << x << ' ';  // 1 4 10 12 9
}