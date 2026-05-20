#include<bits/stdc++.h>
const int MOD = 998244353;
using namespace std;

class Derangement{
private:
    int P;
public:
    vector<int>D;
    Derangement(int n , int p = MOD) : P(p) {
        D.assign(n + 1 , 0);
        D[0] = D[1] = 0; D[2] = 1;
        for(int i=3;i<=n;++i)D[i] = (i  - 1) * (D[i - 1] + D[i - 2]) % P;
    }
};