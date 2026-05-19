#include<bits/stdc++.h>
#define all(x) x.begin() , x.end()
#define rall(x) x.rbegin() , x.rend()
#define ALL(x) x.begin() + 1 , x.end()
#define RALL(x) x.begin() + 1 , x.rend()
using namespace std;
using db = double;
using pdd = pair<db , db>;
const db EPS = 1e-10;
const db PI = acos(-1);

class PT{
public:
    db fs , sc;
    PT(db x = 0 , db y = 0) : fs(x) , sc(y) {}

    friend PT operator+(const PT& a , const PT& b){ return PT(a.fs + b.fs , a.sc + b.sc); }
    friend PT operator-(const PT& a , const PT& b){ return PT(a.fs - b.fs , a.sc - b.sc); }
    friend PT operator*(const db& t , const PT& a){ return PT(t * a.fs , t * a.sc); }
    friend PT operator*(const PT& a , const db& t){ return PT(t * a.fs , t * a.sc); }
    friend db operator*(const PT& a , const PT& b){ return a.fs * b.sc - a.sc * b.fs; } // cross
    friend db operator%(const PT& a , const PT& b){ return a.fs * b.fs + a.sc * b.sc; } // dot

    db len()     { return sqrt(fs * fs + sc * sc); }
    db ang(PT& b){ return acos(*this % b / this -> len() / b.len()); }
    db dis(PT& b){ return sqrt(1.0 * (b.fs - fs) * (b.fs - fs) + (b.sc -sc) * (b.sc - sc)); }
};

PT GetIntsct(PT a , PT DA , PT b , PT DB){ // point ---direction---> 
    db ratio = (a - b) * DB / (DB * DA);
    return a + DA * ratio;
}

void PolarSort(vector<PT>& x){
    sort(all(x) , [&](const PT& a , const PT& b){
        db ag1 = atan2(a.sc , a.fs) , ag2 = atan2(b.sc , b.fs);
        if(ag1 < 0)ag1 += 2 * PI;
        if(ag2 < 0)ag2 += 2 * PI;
        return ag1 < ag2;
    });
}

void PolarSort_CrossProductVer(vector<PT>& x){
    auto half = [](const PT& a) -> int {
        return (a.sc < 0 || (a.sc == 0 && a.fs < 0)) ? 1 : 0;
    };
    sort(all(x) , [&](const PT& a, const PT& b){
        if(half(a) != half(b)) return half(a) < half(b);
        db cross = a * b;
        return cross > EPS;
    });
}