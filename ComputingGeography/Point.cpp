#include<bits/stdc++.h>
using db = double;

class point{
public:
    db fs , sc;
    point(db x , db y) : fs(x) , sc(y) {}

    friend point operator+(const point& a , const point& b){ return point(a.fs + b.fs , a.sc + b.sc); }
    friend point operator-(const point& a , const point& b){ return point(a.fs - b.fs , a.sc - b.sc); }
    friend point operator*(const db& t    , const point& a){ return point(t * a.fs , t * a.sc); }
    friend point operator*(const point& a , const db& t)   { return point(t * a.fs , t * a.sc); }
    friend db    operator*(const point& a , const point& b){ return a.fs * b.sc - a.sc * b.fs; } // cross
    friend db    operator%(const point& a , const point& b){ return a.fs * b.fs + a.sc * b.sc; } // dot

    db len()        { return sqrt(fs * fs + sc * sc); }
    db ang(point& b){ return acos(*this % b / this -> len() / b.len()); }
    db dis(point& b){ return sqrt(1.0 * (b.fs - fs) * (b.fs - fs) + (b.sc -sc) * (b.sc - sc)); }
};

point getIntsct(point a , point DA , point b , point DB){ // point ---direction---> 
    db ratio = (a - b) * DB / (DB * DA);
    return a + DA * ratio;
}