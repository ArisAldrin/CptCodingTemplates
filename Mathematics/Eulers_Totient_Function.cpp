#include<bits/stdc++.h>
#define int long long
using namespace std;

class Phi{
public:
    vector<bool> vs;
    vector<int> pm , pos;
    Phi(int n){
        vs.assign(n + 1 , false); pos.assign(n + 1 , 0);
        pos[1] = 1;
        for(int i=2;i<=n;++i){
            if(!vs[i]){
                pm.push_back(i);
                pos[i] = i - 1;
            }
            for(int j=0;pm[j] * i <= n;++j){
                vs[pm[j] * i] = true;
                if(i % pm[j] == 0){
                    pos[pm[j] * i] =  pos[i] * pm[j];
                    break;
                }else{
                    pos[pm[j] * i] = (pm[j] - 1) * pos[i];
                }
            }
        }
    }
};

int single_phi(int n){
    int res = n;
    for(int i=2;i * i <= n;++i){
        if(n % i == 0){
            res = res / i * (i - 1);
            while(n % i == 0)n /= i;
        }
    }
    if(n > 1)res = res / n * (n - 1);
    return res;
}