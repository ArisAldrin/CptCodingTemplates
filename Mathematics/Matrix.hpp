#include<bits/stdc++.h>
#define int long long
using namespace std;
const int MOD = 998244353;

class Mat{
public:
    int n , m , P;
    vector<vector<int> > pos;
    Mat(const vector<vector<int> > & mat , const int & p = MOD) : n(mat.size()) , m(mat.empty() ? 0 : mat[0].size()) , pos(mat) , P(p) {}
    Mat(const int & n , const int & m , const int & p = MOD) : n(n) , m(m) , P(p) { pos.assign(n , vector<int>(m , 0)); }

    friend Mat operator*(const Mat & A , const Mat & B){
        assert(A.m == B.n && A.P == B.P);
        Mat res(A.n , B.m , A.P);
        for(int i=0;i<A.n;++i){
            for(int j=0;j<B.m;++j){
                for(int k=0;k<A.m;++k){
                    res.pos[i][j] = (res.pos[i][j] + A.pos[i][k] * B.pos[k][j] % A.P) % A.P;
                }
            }
        }
        return res;
    }
    friend Mat operator+(const Mat & A , const Mat & B){
        assert(A.n == B.n && A.m == B.m && A.P == B.P);
        Mat res(A.n, A.m, A.P);
        for (int i=0;i<A.n;++i)for(int j=0;j<A.m;++j)res.pos[i][j] = (A.pos[i][j] + B.pos[i][j]) % A.P;
        return res;
    }
    friend Mat operator*(const Mat & A , const int & x){
        Mat res(A.n , A.m , A.P);
        for(int i=0;i<A.n;++i)for(int j=0;j<A.m;++j)res.pos[i][j] = A.pos[i][j] * x % A.P;
        return res;
    }
    friend Mat operator*(const int & x , const Mat & A){ return A * x; }

    void set_identity(){ assert(n == m); for(int i=0;i<n;++i)pos[i][i] = 1; }
    void set_zeros()   { for(int i=0;i<n;++i)for(int j=0;j<m;++j)pos[i][j] = 0; }
    void mfpow(int x){
        assert(n == m);
        Mat A = *this , res(n , n , P); res.set_identity();
        while(x){
            if(x & 1)res = res * A;
            A = A * A;
            x >>= 1;
        }
        pos = res.pos;
    }
};