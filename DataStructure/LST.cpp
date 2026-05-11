#include<bits/stdc++.h>

#define rerr cerr << "\033[31m"
#define erl "\033[0m" << '\n'
#define NO cout << "NO" << '\n';
#define YES cout << "YES" << '\n';
#define No cout << "No" << '\n';
#define Yes cout << "Yes" << '\n';
#define NOO cout << -1 << '\n';
#define all(x) (x).begin() , (x).end()
#define ALL(X) (X).begin() + 1 , (X).end()
#define rall(x) (x).rbegin() , (x).rend()
#define RALL(X) (X).rbegin() , (X).rend() - 1
#define fs first
#define sc second
#define arr array
#define int long long

using namespace std;
using ull = unsigned long long;
using db = double;
using pii = pair<int , int>;
using piii = arr<int , 3>;
using pdd = pair<db , db>;
using i128 = __int128_t;

const int inf = 1e9 + 7;
const int nil = -inf;
const int INF = 1e18;
const int NIL = -INF;
const int MOD = 998244353;
const db EPS = 1e-10;
const db PI = acos(-1);
mt19937 rdint(time(0));
mt19937_64 rdll(time(0));

struct Tag {
    // Member Variable ...
    Tag() {}
    bool empty() const {
        // How is the Tag empty ...
    }
    void apply(const Tag &t) & {
        // Tag merge ...
    }
};

struct Info {
    // Member Variable ...
    Info() {}
    void apply(const Tag &t) & {
        // How Tag apply to the data ...
    }
};

Info operator+(const Info &a, const Info &b) {
    Info res;
    // Segment Merge ...
    return res;
}

template<class Info, class Tag>
struct LazySegmentTree {
    int n;
    vector<Info> info;
    vector<Tag> tag;

    LazySegmentTree(int n_) : n(n_) {
        info.assign(4 * n + 1, Info());
        tag.assign(4 * n + 1, Tag());
    }

    template<class T>
    void build(const vector<T>& a, int p, int l, int r) {
        if (l == r) {
            info[p] = Info(a[l]);
            return;
        }
        int m = (l + r) / 2;
        build(a, 2 * p, l, m);
        build(a, 2 * p + 1, m + 1, r);
        pull(p);
    }

    void pull(int p) {
        info[p] = info[2 * p] + info[2 * p + 1];
    }

    void apply(int p, const Tag &v) {
        info[p].apply(v);
        tag[p].apply(v);
    }

    void push(int p) {
        if (tag[p].empty()) return;
        apply(2 * p, tag[p]);
        apply(2 * p + 1, tag[p]);
        tag[p] = Tag();
    }

    void modify(int p, int l, int r, int x, const Info &v) {
        if (l == r) {
            info[p] = v;
            return;
        }
        int m = (l + r) / 2;
        push(p);
        if (x <= m) modify(2 * p, l, m, x, v);
        else modify(2 * p + 1, m + 1, r, x, v);
        pull(p);
    }

    Info rangeQuery(int p, int l, int r, int x, int y) {
        if (l >= x && r <= y) return info[p];
        int m = (l + r) / 2;
        push(p);
        if (y <= m) return rangeQuery(2 * p, l, m, x, y);
        if (x > m) return rangeQuery(2 * p + 1, m + 1, r, x, y);
        return rangeQuery(2 * p, l, m, x, y) + rangeQuery(2 * p + 1, m + 1, r, x, y);
    }

    void rangeApply(int p, int l, int r, int x, int y, const Tag &v) {
        if (l >= x && r <= y) {
            apply(p, v);
            return;
        }
        int m = (l + r) / 2;
        push(p);
        if (x <= m) rangeApply(2 * p, l, m, x, y, v);
        if (y > m) rangeApply(2 * p + 1, m + 1, r, x, y, v);
        pull(p);
    }

    void build(const vector<int>& a) { build(a, 1, 1, n); }
    void modify(int x, const Info &v) { modify(1, 1, n, x, v); }
    Info rangeQuery(int l, int r) { return rangeQuery(1, 1, n, l, r); }
    void rangeApply(int l, int r, const Tag &v) { rangeApply(1, 1, n, l, r, v); }
};

void solve(){
    
}

signed main(){
    ios::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);
    cout << fixed << setprecision(15);
    int t;cin >> t;
    while(t --)
    solve();
    return 0;
}