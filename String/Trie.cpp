#include<bits/stdc++.h>
#define int long long
#define arr array
using namespace std;

class Trie{
public:
    vector<arr<int, 26> >ch;
    vector<int> cnt , pass;
    Trie(){
        ch.push_back({});
        ch[0].fill(0);
        cnt.push_back(0);
        pass.push_back(0);
    }

    void insert(const string& s){
        int u = 0;
        pass[u]++;
        for (char c:s){
            int x = c - 'a';
            if(!ch[u][x]){
                ch.push_back({});
                ch.back().fill(0);
                cnt.push_back(0);
                pass.push_back(0);
                ch[u][x] = ch.size() - 1;
            }
            u = ch[u][x];
            pass[u]++;
        }
        cnt[u]++;
    }

    int search(const string& s){
        int u = 0;
        for(char c:s){
            int x = c - 'a';
            if(!ch[u][x]) return 0;
            u = ch[u][x];
        }
        return cnt[u];
    }

    int askpre(const string& s){
        int u = 0;
        for(char c:s){
            int x = c - 'a';
            if(!ch[u][x])return 0;
            u = ch[u][x];
        }
        return pass[u];
    }

    void erase(const string& s){
        if(!search(s))return;
        int u = 0;
        pass[u]--;
        for(char c:s){
            int x = c - 'a';
            u = ch[u][x];
            pass[u]--;
        }
        cnt[u]--;
    }
};

void solve(){
    int n;cin >> n;
    Trie trie;
    while(n--){
        int op; string s;
        cin >> op >> s;
        if(op == 1)trie.insert(s);
        else if(op == 2)trie.erase(s);
        else if(op == 3)cout << (trie.search(s) ? "YES" : "NO") << '\n';
        else cout << trie.askpre(s) << '\n';
    }
}

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

void solve2(){
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
