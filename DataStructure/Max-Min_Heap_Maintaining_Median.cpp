#include<bits/stdc++.h>
using namespace std;

class Max_Min_Heap{
private:
    multiset<int>low , high;
    //low : 1 ~ (n + 1) / 2 , high : (n + 1) / 2 + 1 ~ n
    //median : *low.rbegin();
    void balance(){
        while(low.size() > high.size() + 1){
            high.insert(*low.rbegin());
            low.erase(-- low.end());
        }
        while(low.size() < high.size()){
            low.insert(*high.begin());
            high.erase(high.begin());
        }
    }
public:
    void add(int x){
        if(!high.empty() && x >= *high.begin())high.insert(x);
        else low.insert(x);
        balance();
    }
    void del(int x){
        if(high.count(x))high.erase(high.find(x));
        else if(low.count(x))low.erase(low.find(x));
        balance();
    }
    int get(){
        return *low.rbegin();
    }
};
