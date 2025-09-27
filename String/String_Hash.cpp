#include<bits/stdc++.h>
using namespace std;
using ll = long long;
using ull = unsigned long long;

class StringHash{
private:
    const ll P = 13331;
    vector<ull>poly , hash_val;
public:
    StringHash(string s){
        int n = s.size();
        s = ' ' + s;
        poly.assign(n + 1 , 1); hash_val.assign(n + 1 , 0);
        for(int i=1;i<=n;++i){
            poly[i] = poly[i - 1] * P;
            hash_val[i] = hash_val[i - 1] * P + s[i];
        }
    }
    ull cal(int l , int r){
        return hash_val[r] - hash_val[l - 1] * poly[r - l + 1];
    }
    bool same(int l1 , int r1 , int l2 , int r2){
        return cal(l1 , r1) == cal(l2 , r2);
    }
};
