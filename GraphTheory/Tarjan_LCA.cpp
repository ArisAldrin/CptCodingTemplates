#include<bits/stdc++.h>
#define all(x) (x).begin() , (x).end()
#define fs first
#define sc second
using namespace std;
using pii = pair<int , int>;

int n , q;
vector<vector<int> >g;
vector<vector<pii> >qry;
vector<int>fa , ans;
vector<bool>vis;

int find(int x){
    if(x == fa[x])return x;
    else return fa[x] = find(fa[x]);
}

void dfs(int u){
    vis[u] = true;
    for(auto v:g[u]){
        if(!vis[v]){
            dfs(v);
            fa[v] = u;
        }
    }
    for(auto i:qry[u]){
        int v = i.fs , id = i.sc;
        if(vis[v])ans[id] = find(v);
    }
}

signed main(){
    cin >> n >> q;
    g.assign(n + 1 , vector<int>());
    qry.assign(n + 1  , vector<pii>());
    for(int i=1;i<=n - 1;++i){
        int u , v;cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    for(int i=1;i<=q;++i){
        int u , v;cin >> u >> v;
        qry[u].push_back({v , i});
        qry[v].push_back({u , i});
    }
    ans.assign(q + 1 , 0);
    fa.assign(n + 1  , 0); iota(all(fa) , 0);
    vis.assign(n + 1 , false);
    dfs(1);
    for(int i=1;i<(int)ans.size();++i)cout << ans[i] << '\n';
    return 0;
}