const int MOD = 998244353;

int fpow(int a , int n){
    int res = 1;
    while(n){
        if(n & 1)res *= a;
        a *= a;
        n >>= 1;
    }
    return res;
}

int fpowMOD(int a , int n , int p = MOD){
    int res = 1; a %= p;
    while(n){
        if(n & 1)res = res * a % p;
        a = a * a % p;
        n >>= 1;
    }
    return res;
}

int inv(int x , int p = MOD){
    return fpowMOD(x , p - 2 , p);
}
