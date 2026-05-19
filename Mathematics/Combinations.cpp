#include <bits/stdc++.h>
#include "Fast_Pow.hpp"

#define rerr cerr << "\033[31m"
#define erl "\033[0m" << '\n'
#define NO cout << "NO" << '\n';
#define YES cout << "YES" << '\n';
#define No cout << "No" << '\n';
#define Yes cout << "Yes" << '\n';
#define NOO cout << -1 << '\n';
#define all(x) (x).begin() , (x).end()
#define ALL(X) (X).begin() + 1 , (X).end()
#define rall(x) (x).rbegin() , (x).rend()
#define RALL(X) (X).rbegin() , (X).rend() - 1
#define fs first
#define sc second
#define arr array
#define int long long

using namespace std;
using ull = unsigned long long;
using db = double;
using pii = pair<int , int>;
using piii = arr<int , 3>;
using pdd = pair<db , db>;
using i128 = __int128_t;

const int inf = 1e9 + 7;
const int nil = -inf;
const int INF = 1e18;
const int NIL = -INF;
const db EPS = 1e-10;
const db PI = acos(-1);
mt19937 rdint(time(0));
mt19937_64 rdll(time(0));


class Comb{
public:
    vector<int>fact , invf;
    Comb(int n , int p = MOD){
        fact.assign(n + 1 , 1); invf.assign(n + 1 , 1);
        for(int i=1;i<=n;++i)fact[i] = fact[i - 1] * i % MOD;
        invf[n] = inv(fact[i]);
        for(int i=n - 1;i>=1;--i)invf[i] = invf[i + 1] * (i + 1) % MOD;
    }
    int C(int M , int N){
        
    }
    int A(int M , int N){

    }
    int Q(int M , int N){
        
    }
    int invC(int M , int N){

    }
}

void solve(){
    
}

signed main(){
    ios::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    cout << fixed << setprecision(15);
    solve();
    return 0;
}