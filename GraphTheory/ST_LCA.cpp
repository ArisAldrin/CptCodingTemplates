#include<bits/stdc++.h>
#define int long long
using namespace std;

int n , m;
vector<int> dep;
vector<vector<int> > fa , g;

void dfs(int now , int from){
    dep[now] = dep[from] + 1;
    fa[now][0] = from;
    for(int i=1;i<=25;++i)fa[now][i] = fa[fa[now][i - 1]][i - 1];
    for(auto nx : g[now])if(nx != from)dfs(nx , now);
}

int lca(int u , int  v){
    if(dep[u] < dep[v])swap(u , v);
    for(int i=25;i>=0;--i)if(dep[fa[u][i]] >= dep[v])u = fa[u][i];
    if(u == v)return v;
    for(int i=25;i>=0;--i)if(fa[u][i] != fa[v][i])u = fa[u][i] , v = fa[v][i];
    return fa[v][0];
}

void solve(){
    cin >> n >> m;
    dep.assign(n + 1 , 0); fa.assign(n + 1 , vector<int>(26 , 0)); g.assign(n + 1 , vector<int>());
    for(int i=1;i<=m;++i){
        int u,v;cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    dfs(1 , 0);
    int q;cin >> q;
    while(q --){
        int u,v;cin >> u >> v;
        cout << lca(u , v) << '\n';
    }
}

signed main(){
    ios::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    cout << fixed << setprecision(15);
    solve();
    return 0;
}