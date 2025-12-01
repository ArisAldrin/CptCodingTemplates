#include<bits/stdc++.h>
#define int long long
using namespace std;
using ll = long long;

class BigInt{
private:
    static const int BASE = 100000000;
    static const int WIDTH = 8;
    vector<int>a;
    bool sign;

    // 去除前导零 : 许多东西的依赖
    void trim() {
        while(a.size() > 1 && a.back() == 0) a.pop_back();
        if(a.size() == 1 && a[0] == 0) sign = false;
    }

    // 无符号比较 : 许多东西的依赖
    bool abs_less(const BigInt& b) const {
        if(a.size() != b.a.size()) return a.size() < b.a.size();
        for(int i=a.size() - 1;i>=0;i--){
            if(a[i] != b.a[i]) return a[i] < b.a[i];
        }
        return false;
    }
    
    // 无符号加减法 : 加减法 , 无符号带余数除法的依赖
    BigInt abs_add(const BigInt& b) const {
        BigInt res;
        res.a.clear();
        int cry = 0;
        for(int i=0;i<a.size() || i < b.a.size() || cry;i++){
            if(i < a.size()) cry += a[i];
            if(i < b.a.size()) cry += b.a[i];
            res.a.push_back(cry % BASE);
            cry /= BASE;
        }
        res.trim();
        return res;
    }

    BigInt abs_sub(const BigInt& b) const {
        BigInt res = *this;
        for(int i = 0, brw = 0; i < res.a.size(); i++){
            res.a[i] -= (i < b.a.size() ? b.a[i] : 0) + brw;
            brw = (res.a[i] < 0);
            if (brw) res.a[i] += BASE;
        }
        res.trim();
        return res;
    }

    // 无符号带余数（通过r传出）除法 ： 除法/取模的依赖 ，同时它依赖乘法重载
    BigInt abs_mod_div(const BigInt& b, BigInt& r) const {
        if (b.a.size() == 1 && b.a[0] == 0) {
            r = *this;
            return BigInt(0); 
        }
        BigInt q(0);
        r = *this;
        if(abs_less(b)) {
            r.sign = false;
            return q;
        } 
        BigInt tmpr = *this;
        q.a.resize(a.size() - b.a.size() + 1);
        BigInt cpb = b; 
        cpb.sign = false; 
        for(int i = a.size() - b.a.size(); i >= 0; --i){
            BigInt sfd = cpb;
            sfd.a.insert(sfd.a.begin(), i, 0); 
            ll low = 0, high = BASE - 1, ansq = 0;
            while(low <= high){
                ll mid = low + (high - low) / 2;
                BigInt P = sfd * BigInt(mid); 
                if(!tmpr.abs_less(P)){
                    ansq = mid;
                    low = mid + 1;
                }else{
                    high = mid - 1;
                }
            }
            q.a[i] = (int)ansq;   
            BigInt fnp = sfd * BigInt(ansq); 
            tmpr = tmpr.abs_sub(fnp);
        }
        q.trim();
        r = tmpr;
        r.sign = false;
        return q;
    }

public:
    BigInt(ll x = 0) : sign(false){
        if(x < 0) sign = true, x = -x;
        if(x == 0) a.push_back(0);
        else while(x > 0){
            a.push_back(x % BASE);
            x /= BASE;
        }
        trim();
    }

    BigInt(const string& s) : sign(false){
        int st = 0;
        if(s[0] == '-') sign = true, st = 1;
        
        for(int i=s.length();i>st;i-=WIDTH){
            if(i - WIDTH < st) a.push_back(stoi(s.substr(st, i - st)));
            else a.push_back(stoi(s.substr(i - WIDTH, WIDTH)));
        }
        if(s.length() == st) a.push_back(0);
        trim();
    }

    // <<<<<<<<<<<<<<<<<<<<<< 比较运算符 >>>>>>>>>>>>>>>>>>>>>>
    bool operator<(const BigInt& b) const {
        if(sign != b.sign) return sign;
        if(!sign) return abs_less(b); 
        return b.abs_less(*this);
    }
    bool operator>(const BigInt& b) const { return b < *this; }
    bool operator<=(const BigInt& b) const { return !(*this > b); }
    bool operator>=(const BigInt& b) const { return !(*this < b); }
    bool operator==(const BigInt& b) const { return sign == b.sign && a == b.a; }
    bool operator!=(const BigInt& b) const { return !(*this == b); }

    // +++++++++++++++++++++++++ 加法 +++++++++++++++++++++++++
    BigInt operator+(const BigInt& b) const {
        if(sign == b.sign){
            BigInt res = abs_add(b);
            res.sign = sign;
            return res;
        }else{
            if(abs_less(b)){
                BigInt tmp = b.abs_sub(*this);
                tmp.sign = b.sign;
                return tmp;
            }else{
                BigInt tmp = abs_sub(b);
                tmp.sign = sign;
                return tmp;
            }
        }
    }

    // ---------------------- 减法(依赖加法) ----------------------
    BigInt operator-(const BigInt& b) const {
        BigInt tmp = b;
        tmp.sign = !b.sign;
        return *this + tmp;
    }

    // ********************** 乘法 **********************
    BigInt operator*(const BigInt& b) const {
        BigInt res;
        res.a.assign(a.size() + b.a.size(), 0);
        for(int i=0;i<a.size();++i){
            ll cry = 0;
            for(int j=0;j<b.a.size() || cry;++j){
                ll cur = res.a[i + j] + cry;
                if(j < b.a.size()) cur += (ll)a[i] * b.a[j];
                res.a[i + j] = cur % BASE;
                cry = cur / BASE;
            }
        }
        res.trim();
        res.sign = (res.a.size() != 1 || res.a[0] != 0) && (sign != b.sign);
        return res;
    }

    // ////////////////////// 除法 //////////////////////
    BigInt operator/(const BigInt& b) const {
        BigInt r;
        BigInt q = abs_mod_div(b, r);
        bool signres = (q.a.size() != 1 || q.a[0] != 0) && (sign != b.sign);
        q.sign = signres;
        return q;
    }

    BigInt operator%(const BigInt& b) const {
        BigInt r;
        abs_mod_div(b, r);
        bool signres = (r.a.size() != 1 || r.a[0] != 0) && sign;
        r.sign = signres;
        return r;
    }

    // <<<<<<<<<<<<<<<<<<<<<< 输入输出 >>>>>>>>>>>>>>>>>>>>>>>
    friend istream& operator>>(istream& is, BigInt& n){
        string s;
        is >> s;
        n = BigInt(s);
        return is;
    }
    friend ostream& operator<<(ostream& os, const BigInt& n){
        if(n.a.empty() || (n.a.size() == 1 && n.a[0] == 0)) return os << 0;
        if(n.sign) os << '-';
        os << n.a.back();
        for(int i=n.a.size() - 2;i>=0;--i){
            os << setfill('0') << setw(WIDTH) << n.a[i];
        }
        return os;
    }
};
