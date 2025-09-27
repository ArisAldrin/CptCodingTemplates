#include<bits/stdc++.h>
#define all(x) (x).begin() , (x).end()
using namespace std;
using i128 = __int128_t;

istream &operator>>(istream &is , i128 &n){
    n = 0;
    string s;
    is >> s;
    for(auto c:s)n = 10 * n + c - '0';
    return is;
}

ostream &operator<<(ostream &os , i128 &n){
    if(n == 0)return os << 0;
    string s;
    while(n > 0){
        s += '0' + n % 10;
        n /= 10;
    }
    reverse(all(s));
    return os << s;
}
