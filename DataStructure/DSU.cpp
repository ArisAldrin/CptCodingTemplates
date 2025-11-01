#include<bits/stdc++.h>
#define all(x) (x).begin() , (x).end()
using namespace std; 

class DSU{
private:
    vector<int>fa , rk;
public:
    DSU(int n){
        fa.assign(n , 0); rk.assign(n , 0);
        iota(all(fa) , 0);
    }
    int find(int x){
        if(fa[x] == x)return x;
        else return fa[x] = find(fa[x]);
    }

    void merge(int x , int y){
        x = find(x); y = find(y);
        if(x == y)return;
        if(rk[x] < rk[y]){
            fa[x] = y;
        }else{
            fa[y] = x;
            if(rk[x] == rk[y])rk[x] ++;
        }
    }

    bool same(int x , int y){
        return find(x) == find(y);
    }
};