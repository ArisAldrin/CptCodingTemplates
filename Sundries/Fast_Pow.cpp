int fpow(int a , int n){
    int res = 1;
    while(n){
        if(n & 1)res *= a;
        a *= a;
        n >>= 1;
    }
    return res;
}

int fpowMOD(int a , int n , int p){
    int res = 1;
    while(n){
        if(n & 1)res = res * a % p;
        a = a * a % p;
        n >>= 1;
    }
    return res;
}
