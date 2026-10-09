#include<bits/stdc++.h>
#define int long long
using namespace std;

const int MOD = 998244353;
const int G = 3;
const int Gi = 332748118;
using u128 = unsigned __int128;

int fpowMOD(int a, int n, int p = MOD) {
    int res = 1, b = a % p;
    while (n) {
        if (n & 1) res = res * b % p;
        b = b * b % p;
        n >>= 1;
    }
    return res;
}

void NTT(vector<int>& a, int type) {
    int n = (int)a.size();
    static vector<int> r;
    if ((int)r.size() != n) {
        r.assign(n, 0);
        for (int i = 1; i < n; ++i)
            r[i] = (r[i >> 1] >> 1) | ((i & 1) ? n >> 1 : 0);
    }
    for (int i = 0; i < n; ++i) if (i < r[i]) swap(a[i], a[r[i]]);
    for (int mid = 1; mid < n; mid <<= 1) {
        int wn = fpowMOD(type == 1 ? G : Gi, (MOD - 1) / (mid << 1));
        for (int j = 0; j < n; j += mid << 1) {
            int w = 1;
            for (int k = 0; k < mid; ++k, w = w * wn % MOD) {
                int x = a[j + k];
                int y = (int)(w * a[j + mid + k] % MOD);
                a[j + k] = x + y < MOD ? x + y : x + y - MOD;
                a[j + mid + k] = x - y >= 0 ? x - y : x - y + MOD;
            }
        }
    }
    if (type == -1) {
        int inv = fpowMOD(n, MOD - 2);
        for (int& x : a) x = (int)(1LL * x * inv % MOD);
    }
}

class BigInt {
private:
    static const int BASE = 100000000;
    static const int WIDTH = 8;
    vector<int> a;
    bool sign = false;

    bool is_zero() const { return a.size() == 1 && a[0] == 0; }
    void trim() {
        while (a.size() > 1 && a.back() == 0) a.pop_back();
        if (a.empty()) a.push_back(0);
        if (is_zero()) sign = false;
    }
    bool abs_less(const BigInt& b) const {
        if (a.size() != b.a.size()) return a.size() < b.a.size();
        for (int i = (int)a.size() - 1; i >= 0; --i)
            if (a[i] != b.a[i]) return a[i] < b.a[i];
        return false;
    }
    BigInt abs_add(const BigInt& b) const {
        BigInt z;
        z.a.clear();
        long long carry = 0;
        for (int i = 0; i < (int)max(a.size(), b.a.size()) || carry; ++i) {
            long long cur = carry;
            if (i < (int)a.size()) cur += a[i];
            if (i < (int)b.a.size()) cur += b.a[i];
            z.a.push_back((int)(cur % BASE));
            carry = cur / BASE;
        }
        z.trim();
        return z;
    }
    BigInt abs_sub(const BigInt& b) const {
        BigInt z = *this;
        long long borrow = 0;
        for (int i = 0; i < (int)z.a.size(); ++i) {
            long long cur = (long long)z.a[i] - borrow - (i < (int)b.a.size() ? b.a[i] : 0);
            if (cur < 0) cur += BASE, borrow = 1;
            else borrow = 0;
            z.a[i] = (int)cur;
        }
        z.trim();
        return z;
    }

    BigInt abs_mul_small(int x) const {
        BigInt z;
        z.a.assign(a.size(), 0);
        long long carry = 0;
        for (int i = 0; i < (int)a.size(); ++i) {
            long long cur = 1LL * a[i] * x + carry;
            z.a[i] = (int)(cur % BASE);
            carry = cur / BASE;
        }
        if (carry) z.a.push_back((int)carry);
        z.trim();
        return z;
    }

    int abs_div_small_inplace(int x) {
        long long rem = 0;
        for (int i = (int)a.size() - 1; i >= 0; --i) {
            long long cur = rem * BASE + a[i];
            a[i] = (int)(cur / x);
            rem = cur % x;
        }
        sign = false;
        trim();
        return (int)rem;
    }

    BigInt shl_limbs(int k) const {
        if (is_zero()) return BigInt(0);
        BigInt z = *this;
        z.a.insert(z.a.begin(), k, 0);
        return z;
    }

    BigInt shr_limbs(int k) const {
        if (k >= (int)a.size()) return BigInt(0);
        BigInt z = *this;
        z.a.erase(z.a.begin(), z.a.begin() + k);
        z.trim();
        return z;
    }

    static BigInt base_power(int k) {
        BigInt z;
        z.a.assign(k + 1, 0);
        z.a[k] = 1;
        return z;
    }

    static BigInt from_u128(u128 x) {
        BigInt z;
        z.a.clear();
        do {
            z.a.push_back((int)(x % BASE));
            x /= BASE;
        } while (x != 0);
        return z;
    }

    static BigInt approximate_reciprocal(const BigInt& A) {
        const int n = (int)A.a.size();
        assert(!A.sign && A.a.back() >= BASE / 2);

        if (n <= 2) {
            u128 value = 0;
            for (int i = n - 1; i >= 0; --i)
                value = value * BASE + A.a[i];
            u128 power = 1;
            for (int i = 0; i < 2 * n; ++i) power *= BASE;
            return from_u128((power - 1) / value);
        }

        const int low = (n - 1) / 2;
        const int high = n - low;
        BigInt Ah = A.shr_limbs(low);
        BigInt Xh = approximate_reciprocal(Ah);

        BigInt T = A * Xh;
        BigInt P = base_power(n + high);
        while (T >= P) {
            Xh = Xh - BigInt(1);
            T = T - A;
        }

        T = P - T;
        BigInt Tm = T.shr_limbs(low);
        BigInt U = Tm * Xh;
        BigInt correction = U.shr_limbs(2 * high - low);
        return Xh.shl_limbs(low) + correction;
    }

    BigInt divmod_knuth_abs(const BigInt& divisor0, BigInt& remainder) const {
        BigInt dividend = *this, divisor = divisor0;
        dividend.sign = divisor.sign = false;
        if (dividend.abs_less(divisor)) { remainder = dividend; return BigInt(0); }
        if (divisor.a.size() == 1) {
            BigInt q = dividend;
            remainder = BigInt(q.abs_div_small_inplace(divisor.a[0]));
            return q;
        }
        int n = (int)dividend.a.size(), m = (int)divisor.a.size();
        int norm = BASE / (divisor.a.back() + 1);
        BigInt u = dividend.abs_mul_small(norm), v = divisor.abs_mul_small(norm);
        u.a.resize(n + 1, 0);
        BigInt q;
        q.a.assign(n - m + 1, 0);
        for (int j = n - m; j >= 0; --j) {
            long long top = 1LL * u.a[j + m] * BASE + u.a[j + m - 1];
            long long qhat = top / v.a[m - 1], rhat = top % v.a[m - 1];
            if (qhat >= BASE) {
                qhat = BASE - 1;
                rhat = top - qhat * v.a[m - 1];
            }
            while (qhat * v.a[m - 2] > rhat * BASE + u.a[j + m - 2]) {
                --qhat;
                rhat += v.a[m - 1];
                if (rhat >= BASE) break;
            }
            long long borrow = 0;
            for (int i = 0; i < m; ++i) {
                long long cur = (long long)u.a[j + i] - qhat * v.a[i] - borrow;
                if (cur < 0) {
                    borrow = (-cur + BASE - 1) / BASE;
                    cur += borrow * BASE;
                } else borrow = 0;
                u.a[j + i] = (int)cur;
            }
            long long hi = (long long)u.a[j + m] - borrow;
            while (hi < 0) {
                --qhat;
                long long carry = 0;
                for (int i = 0; i < m; ++i) {
                    long long cur = (long long)u.a[j + i] + v.a[i] + carry;
                    if (cur >= BASE) cur -= BASE, carry = 1;
                    else carry = 0;
                    u.a[j + i] = (int)cur;
                }
                hi += carry;
            }
            u.a[j + m] = (int)hi;
            q.a[j] = (int)qhat;
        }
        q.trim();
        remainder.a.assign(u.a.begin(), u.a.begin() + m);
        remainder.sign = false;
        remainder.trim();
        remainder.abs_div_small_inplace(norm);
        return q;
    }

    BigInt divmod_newton_abs(const BigInt& divisor0, BigInt& remainder) const {
        BigInt dividend = *this, divisor = divisor0;
        dividend.sign = divisor.sign = false;
        if (dividend.abs_less(divisor)) { remainder = dividend; return BigInt(0); }

        int norm = BASE / (divisor.a.back() + 1);
        BigInt A = dividend.abs_mul_small(norm);
        BigInt B = divisor.abs_mul_small(norm);
        int p = (int)A.a.size(), n = (int)B.a.size();

        BigInt padded_B = B.shl_limbs(p - n);
        BigInt inverse = approximate_reciprocal(padded_B);
        BigInt quotient = (A * inverse).shr_limbs(p + n);
        BigInt rem = A - quotient * B;

        while (rem.sign) {
            quotient = quotient - BigInt(1);
            rem = rem + B;
        }
        while (rem >= B) {
            quotient = quotient + BigInt(1);
            rem = rem - B;
        }
        rem.abs_div_small_inplace(norm);
        remainder = rem;
        quotient.trim();
        return quotient;
    }

    BigInt divmod_abs(const BigInt& b, BigInt& r, bool force_newton = false) const {
        if (b.is_zero()) throw domain_error("division by zero");
        BigInt x = *this, y = b;
        x.sign = y.sign = false;
        if (x.abs_less(y)) { r = x; return BigInt(0); }
        int qn = (int)x.a.size() - (int)y.a.size() + 1;
        if (!force_newton && min(qn, (int)y.a.size()) < 64)
            return x.divmod_knuth_abs(y, r);
        return x.divmod_newton_abs(y, r);
    }

public:
    BigInt(long long x = 0) {
        sign = x < 0;
        unsigned long long y = sign ? 0ULL - (unsigned long long)x : (unsigned long long)x;
        do { a.push_back((int)(y % BASE)); y /= BASE; } while (y);
        trim();
    }
    BigInt(const string& s) {
        int st = 0;
        if (!s.empty() && s[0] == '-') sign = true, st = 1;
        for (int i = (int)s.size(); i > st; i -= WIDTH) {
            int l = max(st, i - WIDTH);
            a.push_back(stoi(s.substr(l, i - l)));
        }
        if (a.empty()) a.push_back(0);
        trim();
    }
    bool operator<(const BigInt& b) const {
        if (sign != b.sign) return sign;
        return sign ? b.abs_less(*this) : abs_less(b);
    }
    bool operator>(const BigInt& b) const { return b < *this; }
    bool operator<=(const BigInt& b) const { return !(b < *this); }
    bool operator>=(const BigInt& b) const { return !(*this < b); }
    bool operator==(const BigInt& b) const { return sign == b.sign && a == b.a; }
    bool operator!=(const BigInt& b) const { return !(*this == b); }
    BigInt operator+(const BigInt& b) const {
        if (sign == b.sign) { BigInt z = abs_add(b); z.sign = sign && !z.is_zero(); return z; }
        if (abs_less(b)) { BigInt z = b.abs_sub(*this); z.sign = b.sign && !z.is_zero(); return z; }
        BigInt z = abs_sub(b); z.sign = sign && !z.is_zero(); return z;
    }
    BigInt operator-(const BigInt& b) const {
        BigInt t = b;
        if (!t.is_zero()) t.sign = !t.sign;
        return *this + t;
    }
    BigInt operator*(const BigInt& b) const {
        if (is_zero() || b.is_zero()) return BigInt(0);
        if (a.size() + b.a.size() < 64) {
            BigInt z;
            z.a.assign(a.size() + b.a.size(), 0);
            for (int i = 0; i < (int)a.size(); ++i) {
                long long carry = 0;
                for (int j = 0; j < (int)b.a.size() || carry; ++j) {
                    long long cur = z.a[i + j] + carry;
                    if (j < (int)b.a.size()) cur += 1LL * a[i] * b.a[j];
                    z.a[i + j] = (int)(cur % BASE);
                    carry = cur / BASE;
                }
            }
            z.sign = sign != b.sign;
            z.trim();
            return z;
        }
        vector<int> x, y;
        x.reserve(a.size() * WIDTH);
        y.reserve(b.a.size() * WIDTH);
        for (int limb : a) for (int i = 0; i < WIDTH; ++i) x.push_back(limb % 10), limb /= 10;
        for (int limb : b.a) for (int i = 0; i < WIDTH; ++i) y.push_back(limb % 10), limb /= 10;
        int need = (int)x.size() + (int)y.size() - 1, len = 1;
        while (len < need) len <<= 1;
        x.resize(len); y.resize(len);
        NTT(x, 1); NTT(y, 1);
        for (int i = 0; i < len; ++i) x[i] = (int)(1LL * x[i] * y[i] % MOD);
        NTT(x, -1);
        vector<int> decimal;
        long long carry = 0;
        for (int i = 0; i < need || carry; ++i) {
            if (i < need) carry += x[i];
            decimal.push_back((int)(carry % 10));
            carry /= 10;
        }
        while (decimal.size() > 1 && decimal.back() == 0) decimal.pop_back();
        BigInt z;
        z.a.clear();
        for (int i = 0; i < (int)decimal.size(); i += WIDTH) {
            int limb = 0, p10 = 1;
            for (int j = i; j < min(i + WIDTH, (int)decimal.size()); ++j) {
                limb += decimal[j] * p10;
                p10 *= 10;
            }
            z.a.push_back(limb);
        }
        z.sign = sign != b.sign;
        z.trim();
        return z;
    }
    pair<BigInt, BigInt> divmod(const BigInt& b) const {
        BigInt r;
        BigInt q = divmod_abs(b, r);
        q.sign = !q.is_zero() && (sign != b.sign);
        r.sign = !r.is_zero() && sign;
        return {q, r};
    }
    BigInt operator/(const BigInt& b) const { return divmod(b).first; }
    BigInt operator%(const BigInt& b) const { return divmod(b).second; }
    friend istream& operator>>(istream& is, BigInt& x) {
        string s;
        if (is >> s) x = BigInt(s);
        return is;
    }
    friend ostream& operator<<(ostream& os, const BigInt& x) {
        if (x.sign) os << '-';
        os << x.a.back();
        for (int i = (int)x.a.size() - 2; i >= 0; --i) os << setw(WIDTH) << setfill('0') << x.a[i];
        return os;
    }
};

int main() {
    BigInt a, b;
    cin >> a >> b;
    cout << a / b << ' ' << a % b << '\n';
}