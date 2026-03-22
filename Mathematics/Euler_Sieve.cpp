#include<bits/stdc++.h>
using namespace std;

class Sieve{
public:
    vector<bool>vs;
    vector<int>pm;
    Sieve(int n){
        vs.assign(n + 1 , false);
        for(int i=2;i<=n;++i){
            if(!vs[i])pm.push_back(i);
            for(int j=0;j<(int)pm.size() && pm[j] <= n / i;++j){
                vs[pm[j] * i] = true;
                if(i % pm[j] == 0)break;
            }
        }
    }
    bool is_prime(int x){
        return x > 2 && !vs[x];
    }
};

