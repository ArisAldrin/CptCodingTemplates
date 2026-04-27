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
    BIT(int sz) : n(sz) , tr(sz + 1 , 0) {}
    void add(int id , int x){
        while(id <= n)tr[id] += x , id += lowbit(id);
    }
    int askpre(int id){
        int res = 0;
        while(id)res += tr[id] , id -= lowbit(id);
        return res;
    }
};

class CompleteBIT{
private:
    int n;
    BIT tr1 , tr2;
    int lowbit(int x){ return x & -x; }
    void add(int id , int x){
        tr1.add(id , x);
        tr2.add(id , x * id);
    }
public:
    CompleteBIT(int n) : tr1(n) , tr2(n) , n(n){}
    CompleteBIT(vector<int>& a) : n(a.size() - 1) , tr1(n) , tr2(n){ for(int i=1;i<=n;++i)addrg(i , i , a[i]); }
    
    void addrg(int l , int r , int x){
        add(l , x) , add(r + 1 , -x);
    }

    int askpre(int id){
        return (id + 1) * tr1.askpre(id) - tr2.askpre(id);
    }

    int askrg(int l , int r){
        return askpre(r) - askpre(l - 1);
    }
};