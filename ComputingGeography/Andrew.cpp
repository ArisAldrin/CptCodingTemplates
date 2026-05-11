#include "Point.hpp"

vector<PT> andrew(vector<PT>& ps){
    sort(all(ps) , [](PT& a , PT& b){
        if(a.fs == b.fs)return a.sc < b.sc;
        else return a.fs < b.fs;
    });
    vector<PT>res;
    for(int i=0;i<(int)ps.size();++i){
        while(res.size() >= 2 && (res[res.size() - 1] - res[res.size() - 2]) * (ps[i] - res[res.size() - 2]) <= 0)res.pop_back();
        res.push_back(ps[i]);
    }
    int sz = res.size();
    for(int i=ps.size() - 2;i>=0;--i){
        while(res.size() >= sz + 1 && (res[res.size() - 1] - res[res.size() - 2]) * (ps[i] - res[res.size() - 2]) <= 0)res.pop_back();
        res.push_back(ps[i]);
    }
    res.pop_back(); // delete the start
    return res;
}