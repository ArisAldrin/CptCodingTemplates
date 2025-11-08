#include<bits/stdc++.h>
using namespace std;

#define NO cout<<"NO"<<'\n';
#define NOO cout<<-1<<'\n';
#define YES cout<<"YES"<<'\n';
#define Yes cout<<"Yes"<<'\n';
#define No cout<<"No"<<'\n';
#define fs first
#define sc second

using db=double;
using ll=long long;
using ull=unsigned long long;
using pii=pair<int,int>;
using pll=pair<ll,ll>;
using pdd=pair<db,db>;
using i128=__int128_t;
using ui128=__uint128_t;

const db eps=1e-10;
const db pi=acos(-1.0);
const int inf=1e9+7;
const int nil=-inf;
const ll INF=1e18+10;
const ll NIL=-INF;
const int MOD=998244353;

class KMP{
private:
    string text;
    vector<int> getnx(string p){
        int n=p.size();
        p=' '+p; vector<int>nx(p.size());
        nx[1]=0;
        for(int i=2,j=0;i<=n;++i){
            while(j && p[i]!=p[j+1])j=nx[j];
            if(p[i]==p[j+1])j++;
            nx[i]=j;
        }
        return nx;
    }
public:
    KMP(string s){
        text=s; text=' '+text;
    }
    vector<int> match(string p){
        int n=p.size(),m=text.size();
        vector<int>nx=getnx(p);
        p=' '+p;
        vector<int>pos;
        for(int i=1,j=0;i<=m;++i){
            while(j && text[i]!=p[j+1])j=nx[j];
            if(text[i]==p[j+1])j++;
            if(j==n)pos.push_back(i-n+1);
        }
        return pos;
    }
};