#include<bits/stdc++.h>
using namespace std;
using db = double;
const db EPS = 1e-10;

class DB{
public:
    db v;
    DB(db _v = 0) : v(_v) {}

    bool operator==(const DB& b) const { return abs(v - b.v) < EPS; }
    bool operator< (const DB& b) const { return v < b.v - EPS; }
    bool operator> (const DB& b) const { return v > b.v + EPS; }
    bool operator<=(const DB& b) const { return v < b.v + EPS; }
    bool operator>=(const DB& b) const { return v > b.v - EPS; }

    bool operator==(const db& b) const { return abs(v - b) < EPS; }
    bool operator< (const db& b) const { return v < b - EPS; }
    bool operator> (const db& b) const { return v > b + EPS; }
    bool operator<=(const db& b) const { return v < b + EPS; }
    bool operator>=(const db& b) const { return v > b - EPS; }
};