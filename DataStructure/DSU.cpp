#include<bits/stdc++.h>
#define all(x) (x).begin() , (x).end()
using namespace std; 
using pii = pair<int , int>;
#define fs first
#define sc second

class DSU{
private:
    vector<int>fa , sz;
public:
    DSU(int n){
        fa.assign(n + 1 , 0); sz.assign(n + 1 , 1);
        iota(all(fa) , 0);
    }
    int find(int x){
        return fa[x] == x ? x : fa[x] = find(fa[x]);
    }
    bool same(int x , int y){
        return find(x) == find(y);
    }
    void merge(int x , int y){
        x = find(x); y = find(y);
        if(x == y)return;
        if(sz[x] < sz[y])swap(x , y);
        fa[y] = x;
        sz[x] += sz[y];
    }
    int size(int x){
        return sz[find(x)];
    }
};

class DSU_Rollbackable{
private:
    vector<int>fa , sz;
    vector<pii>st;
    DSU_Rollbackable(int n){
        fa.assign(n + 1 , 0);
        sz.assign(n + 1 , 1);
        iota(all(fa) , 0);
    }
    int find(int x){
        while(fa[x] != x)x = fa[x];
        return x;
    }
    void merge(int x , int y){
        x = find(x); y = find(y);
        if(x == y)return;
        if(sz[x] < sz[y])swap(x , y);
        st.push_back({y , x});
        sz[x] += sz[y];
        fa[y] = x;
    }
    int cur(){
        return st.size();
    }
    void rollback(int k){
        while(st.size() > k){
            int y = st.back().fs;
            int x = st.back().sc;
            st.pop_back();
            fa[y] = y;
            sz[x] -= sz[y];
        }
    }
};