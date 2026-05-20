#include<bits/stdc++.h>
#define int long long
using namespace std;
const int MOD = 998244353;

class StirlingCycle{
private:
    int P;
public:
    vector<vector<int> > S;
    StirlingCycle(int n , int m , int p = MOD) : P(p) { // long long overflow when n > 20 without P
        S.assign(n + 1 , vector<int>(m + 1 , 0));
        for(int i=0;i<=n;++i)S[i][0] = (i == 0);
        for(int i=1;i<=n;++i)for(int j=1;j<=min(i , m);++j)S[i][j] = (S[i - 1][j - 1] % P + (i - 1) * S[i - 1][j] % P) % P;
    }
};


class StirlingSubset{
private:
    int P;
public:
    vector<vector<int> > S;
    StirlingSubset(int n , int m , int p = MOD) : P(p) { // long long overflow when n > 25 without P
        S.assign(n + 1 , vector<int>(m + 1 , 0));
        for(int i=0;i<=n;++i)S[i][0] = (i == 0);
        for(int i=1;i<=n;++i)for(int j=1;j<=min(m , i);++j)S[i][j] = (S[i - 1][j - 1] % P + S[i - 1][j] * j % P) % P;
    }
};