#include<bits/stdc++.h>
#define int long long
using namespace std;

class TrieNode{
public:
    unordered_map<char , TrieNode*>child;
    // map<char,TrieNode*>child;
    int cnt , cntpre;
    bool isend;
    TrieNode(){cnt = 0; cntpre = 0; isend = false;}
};

class Generic_Trie{
private:
    TrieNode* rt;
public:
    Generic_Trie(){
        rt = new TrieNode();
    }
    void insert(string s){
        TrieNode* now = rt;
        for(auto i:s){
            now -> cntpre++;
            if(now -> child.find(i) == now -> child.end()){
                now -> child[i] = new TrieNode();
            }
            now = now -> child[i];
        }
        now -> cnt++; now -> cntpre++; now -> isend = true;
    }
    int askpre(string s){
        TrieNode* now = rt;
        for(auto i:s){
            if(now -> child.find(i) == now -> child.end())return 0;
            now = now -> child[i];
        }
        return now -> cntpre;
    }
    int find(string s){
        TrieNode* now = rt;
        for(auto i:s){
            if(now -> child.find(i) == now -> child.end())return false;
            now = now -> child[i];
        }
        return now -> isend;
    }
};

void solve(){
    Generic_Trie trie;
    int n;cin >> n;
    for(int i=0;i<n;++i){
        string s;cin >> s;
        trie.insert(s);
    }
    int q;cin >> q;
    while(q --){
        string x;cin >> x;
        if(trie.find(x))cout << "Exist" << '\n';
        else cout << "Nope" << '\n';
    }
    cin >> q;
    while(q--){
        string x;cin >> x;
        cout << trie.askpre(x) << '\n';
    }
}

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    solve();
    return 0;
}
