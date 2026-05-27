#include<bits/stdc++.h>
#define NOO cout << -1 << '\n';
#define all(x) (x).begin() , (x).end()
#define fs first
#define sc second
#define int long long
using namespace std;
using pii = pair<int , int>;

int n,m;
vector<vector<pii> > g; // second : index of edge
vector<int> deg , ans;
vector<bool>vs;

void dfs(int now){
    for(auto &i:g[now]){
        int nx = i.fs , eid = i.sc;
        if(vs[eid])continue;
        vs[eid] = true;
        dfs(nx);
    }
    ans.push_back(now);
}

void solve(){
    cin >> n >> m;
    g.assign(n + 1 , vector<pii>()); vs.assign(m + 1 , false); deg.assign(n + 1 , 0);
    for(int i=1;i<=m;++i){
        int u,v;cin >> u >> v;
        g[u].push_back({v , i});
        g[v].push_back({u , i});
        deg[u] ++  , deg[v] ++;
    }
    for(int i=1;i<=n;++i)if(deg[i] & 1){NOO return;}
    dfs(1);
    if(ans.size() != m + 1){NOO return;}
    reverse(all(ans));
    for(auto i:ans)cout << i << ' '; cout << '\n';
}

signed main(){
    ios::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    cout << fixed << setprecision(15);
    solve();
    return 0;
}