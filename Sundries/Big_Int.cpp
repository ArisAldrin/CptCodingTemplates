#include<bits/stdc++.h>
using namespace std;

const int MOD = 998244353;
const int G = 3;
const int Gi = 332748118;

int qpow(int a, int b) {
    int res = 1; a %= MOD;
    while (b) {
        if (b & 1) res = res * a % MOD;
        a = a * a % MOD;
        b >>= 1;
    }
    return res;
}

void ntt(vector<int>& a, int n, int type) {
    static vector<int> r;
    if (r.size() != n) {
        r.resize(n);
        for (int i = 0; i < n; i++) r[i] = (r[i >> 1] >> 1) | ((i & 1) ? (n >> 1) : 0);
    }
    for (int i = 0; i < n; i++) if (i < r[i]) swap(a[i], a[r[i]]);
    for (int mid = 1; mid < n; mid <<= 1) {
        int wn = qpow(type == 1 ? G : Gi, (MOD - 1) / (mid << 1));
        for (int j = 0; j < n; j += (mid << 1)) {
            int w = 1;
            for (int k = 0; k < mid; k++, w = w * wn % MOD) {
                int x = a[j + k], y = w * a[j + mid + k] % MOD;
                a[j + k] = (x + y) % MOD;
                a[j + mid + k] = (x - y + MOD) % MOD;
            }
        }
    }
    if (type == -1) {
        int inv = qpow(n, MOD - 2);
        for (int i = 0; i < n; i++) a[i] = a[i] * inv % MOD;
    }
}

class BigInt {
private:
    static const int BASE = 100000000;
    static const int WIDTH = 8;
    vector<int> a;
    bool sign;

    void trim() {
        while (a.size() > 1 && a.back() == 0) a.pop_back();
        if (a.empty()) a.push_back(0);
        if (a.size() == 1 && a[0] == 0) sign = false;
    }

    bool abs_less(const BigInt& b) const {
        if (a.size() != b.a.size()) return a.size() < b.a.size();
        for (int i = a.size() - 1; i >= 0; i--)
            if (a[i] != b.a[i]) return a[i] < b.a[i];
        return false;
    }

    BigInt abs_add(const BigInt& b) const {
        BigInt res; res.a.clear();
        int c = 0;
        for (int i = 0; i < max(a.size(), b.a.size()) || c; i++) {
            if (i < a.size()) c += a[i];
            if (i < b.a.size()) c += b.a[i];
            res.a.push_back(c % BASE);
            c /= BASE;
        }
        res.trim();
        return res;
    }

    BigInt abs_sub(const BigInt& b) const {
        BigInt res = *this;
        for (int i = 0, brw = 0; i < res.a.size(); i++) {
            res.a[i] -= (i < b.a.size() ? b.a[i] : 0) + brw;
            brw = (res.a[i] < 0);
            if (brw) res.a[i] += BASE;
        }
        res.trim();
        return res;
    }

    BigInt abs_mod_div(const BigInt& b, BigInt& r) const {
        if (b.a.size() == 1 && b.a[0] == 0) { r = *this; return BigInt(0); }
        BigInt q(0); r = *this;
        if (abs_less(b)) { r.sign = false; return q; }
        q.a.resize(a.size() - b.a.size() + 1);
        for (int i = a.size() - b.a.size(); i >= 0; i--) {
            BigInt sfd = b;
            sfd.a.insert(sfd.a.begin(), i, 0);
            int low = 0, high = BASE - 1, ansq = 0;
            while (low <= high) {
                int mid = low + (high - low) / 2;
                if (!r.abs_less(sfd * BigInt(mid))) { ansq = mid; low = mid + 1; }
                else high = mid - 1;
            }
            q.a[i] = ansq;
            r = r.abs_sub(sfd * BigInt(ansq));
        }
        q.trim(); r.trim(); r.sign = false;
        return q;
    }

public:
    BigInt(int x = 0) : sign(x < 0) {
        x = abs(x);
        if (x == 0) a.push_back(0);
        while (x > 0) { a.push_back(x % BASE); x /= BASE; }
        trim();
    }
    BigInt(const string& s) : sign(false) {
        int st = (s[0] == '-' ? 1 : 0);
        if (s[0] == '-') sign = true;
        for (int i = s.length(); i > st; i -= WIDTH) {
            if (i - WIDTH < st) a.push_back(stoll(s.substr(st, i - st)));
            else a.push_back(stoll(s.substr(i - WIDTH, WIDTH)));
        }
        trim();
    }

    bool operator<(const BigInt& b) const {
        if (sign != b.sign) return sign;
        return sign ? b.abs_less(*this) : abs_less(b);
    }
    bool operator>(const BigInt& b) const { return b < *this; }
    bool operator<=(const BigInt& b) const { return !(*this > b); }
    bool operator>=(const BigInt& b) const { return !(*this < b); }
    bool operator==(const BigInt& b) const { return sign == b.sign && a == b.a; }
    bool operator!=(const BigInt& b) const { return !(*this == b); }

    BigInt operator+(const BigInt& b) const {
        if (sign == b.sign) { BigInt res = abs_add(b); res.sign = sign; return res; }
        if (abs_less(b)) { BigInt res = b.abs_sub(*this); res.sign = b.sign; return res; }
        BigInt res = abs_sub(b); res.sign = sign; return res;
    }
    BigInt operator-(const BigInt& b) const {
        BigInt t = b; t.sign = !b.sign; return *this + t;
    }

    BigInt operator*(const BigInt& b) const {
        if ((a.size() == 1 && a[0] == 0) || (b.a.size() == 1 && b.a[0] == 0)) return BigInt(0);
        if (a.size() + b.a.size() < 64) {
            BigInt res; res.a.assign(a.size() + b.a.size(), 0);
            for (int i = 0; i < a.size(); i++) {
                int c = 0;
                for (int j = 0; j < b.a.size() || c; j++) {
                    int cur = res.a[i + j] + c + (j < b.a.size() ? a[i] * b.a[j] : 0);
                    res.a[i + j] = cur % BASE; c = cur / BASE;
                }
            }
            res.sign = sign != b.sign; res.trim(); return res;
        }
        vector<int> va, vb;
        for (int x : a) { for (int i = 0; i < WIDTH; i++) { va.push_back(x % 10); x /= 10; } }
        for (int x : b.a) { for (int i = 0; i < WIDTH; i++) { vb.push_back(x % 10); x /= 10; } }
        int n = 1, m = va.size() + vb.size() - 1;
        while (n <= m) n <<= 1;
        va.resize(n); vb.resize(n);
        ntt(va, n, 1); ntt(vb, n, 1);
        for (int i = 0; i < n; i++) va[i] = va[i] * vb[i] % MOD;
        ntt(va, n, -1);
        BigInt res; res.a.clear();
        int c = 0;
        for (int i = 0; i < m || c; i++) {
            if (i < m) c += va[i];
            res.a.push_back(c % 10); c /= 10;
        }
        string s = ""; if (sign != b.sign) s += '-';
        for (int i = res.a.size() - 1; i >= 0; i--) s += (char)(res.a[i] + '0');
        return BigInt(s);
    }

    BigInt operator/(const BigInt& b) const {
        BigInt r; BigInt q = abs_mod_div(b, r);
        q.sign = (q.a.size() > 1 || q.a[0] > 0) && (sign != b.sign);
        return q;
    }
    BigInt operator%(const BigInt& b) const {
        BigInt r; abs_mod_div(b, r);
        r.sign = (r.a.size() > 1 || r.a[0] > 0) && sign;
        return r;
    }

    friend istream& operator>>(istream& is, BigInt& n) {
        string s; if (!(is >> s)) return is;
        n = BigInt(s); return is;
    }
    friend ostream& operator<<(ostream& os, const BigInt& n) {
        if (n.sign) os << '-';
        os << n.a.back();
        for (int i = n.a.size() - 2; i >= 0; i--) os << setfill('0') << setw(WIDTH) << n.a[i];
        return os;
    }
};

signed main() {
    BigInt a,b;cin >> a >> b;
    cout << a * b << '\n';
    return 0;
}