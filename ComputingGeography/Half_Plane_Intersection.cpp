#include "Point.hpp"

class LN{
public:
    PT p , d;
    db ag;
    LN() {}
    LN(PT p , PT d) : p(p) , d(d) { ag = atan2(d.sc , d.fs); }
    bool operator<(const LN& b) const {
        if(fabs(ag - b.ag) > EPS)return ag < b.ag;
        return d * (b.p - p) > EPS;
    }
};

bool check(LN a , LN b , LN c){
    PT p = GetIntsct(a.p , a.d , b.p , b.d);
    return c.d * (p - c.p) < -EPS;
}

vector<PT> HalfPlaneIntersecion(vector<LN>& L){
    sort(all(L));
    deque<LN>q;
    for(int i=0;i<(int)L.size();++i){
        if(i < (int)L.size() - 1 && fabs(L[i].ag - L[i + 1].ag) < EPS)continue;
        while(q.size() > 1 && check(q.back() , q[q.size() - 2] , L[i]))q.pop_back();
        while(q.size() > 1 && check(q.front() , q[1] , L[i]))q.pop_front();
        q.push_back(L[i]);
    }
    while(q.size() > 1 && check(q.back() , q[q.size() - 2] , q.front()))q.pop_back();
    while(q.size() > 1 && check(q.front() , q[1] , q.back()))q.pop_front();
    if(q.size() < 3) return {};
    vector<PT> ans;
    for(int i=0;i<(int)q.size() - 1;++i)ans.push_back(GetIntsct(q[i].p, q[i].d , q[i + 1].p , q[i + 1].d));
    ans.push_back(GetIntsct(q.back().p , q.back().d , q.front().p , q.front().d));
    return ans;
}