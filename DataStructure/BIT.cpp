#include<bits/stdc++.h>
#define int long long
using namespace std;

class BIT{
private:
    int n;
    vector<int>tr;
    int lowbit(int x){ return x & -x; }
public:
    BIT(vector<int>a){
        tr.assign(a.size() , 0);
        n = a.size() - 1;
        for(int i=1;i<=n;++i)add(i , a[i]);
    }
    void add(int id , int x){
        while(id <= n)tr[id] += x , id += lowbit(id);
    }
    int askpre(int id){
        int res = 0;
        while(id)res += tr[id] , id -= lowbit(id);
        return res;
    }
};