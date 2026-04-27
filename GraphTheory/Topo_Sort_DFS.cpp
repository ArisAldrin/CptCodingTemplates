#include<bits/stdc++.h>
#define NO cout << "NO" << '\n';
#define all(x) x.begin() , x.end()
#define int long long
using namespace std;

vector<vector<int> > g;
vector<int>res , cond;
bool cyc = false;

void dfs(int now){
    if(cyc)return;
    cond[now] = 1;
    for(auto i:g[now]){
        if(cond[i] == 0)dfs(i);
        else if(cond[i] == 1){
            cyc = true;
            return;
        }
    }
    cond[now] = 2;
    res.push_back(now);
}

signed main(){
    int n,m;cin >> n >> m;
    g.assign(n + 1 , vector<int>());
    cond.assign(n + 1 , 0);
    res.clear();
    for(int i=1;i<=m;++i){
        int u,v;cin >> u >> v;
        g[u].push_back(v);
    }
    for(int i=1;i<=n;++i)if(cond[i] == 0)dfs(i);
    reverse(all(res));
    if(!cyc){
        for(auto i:res)cout << i << ' '; cout << '\n';
    }else NO
    return 0;
}