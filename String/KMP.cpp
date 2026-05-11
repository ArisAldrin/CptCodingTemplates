#include<bits/stdc++.h>
using namespace std;

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