#include<bits/stdc++.h>
#define NO cout << "NO" << '\n';
#define int long long
using namespace std;

signed main(){
    int n,m;cin >> n >> m;
    vector<vector<int> > g(n + 1 , vector<int>());
    vector<int>in(n + 1 , 0);
    for(int i=1;i<=m;++i){
        int u,v;cin >> u >> v;
        g[u].push_back(v);
        in[v] ++;
    }
    queue<int>q;
    vector<int>res;
    for(int i=1;i<=n;++i)if(in[i] == 0)q.push(i);
    while(!q.empty()){
        int now = q.front(); q.pop(); res.push_back(now);
        for(auto i:g[now]){
            in[i] --;
            if(in[i] == 0){
                q.push(i);
            }
        }
    }
    if(res.size() == n){
        for(auto i:res)cout << i << ' '; cout << '\n';
    }else NO
    return 0;
}