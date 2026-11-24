// files.h - reading Brain's output files (meta.txt, params.txt, *.bin)
#pragma once
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using KV = std::map<std::string, double>;

// "key value" per line
inline KV read_kv(const std::string& path) {
    KV d;
    std::ifstream f(path);
    std::string k; double v;
    while (f >> k >> v) d[k] = v;
    return d;
}

inline double kv(const KV& m, const std::string& key, double fallback = NAN) {
    auto it = m.find(key);
    return it == m.end() ? fallback : it->second;
}

inline bool file_exists(const std::string& path) { return std::ifstream(path).good(); }

// whole binary file as an array of T (empty if missing)
template <class T> std::vector<T> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    const size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    std::vector<T> v(bytes / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(v.size() * sizeof(T)));
    return v;
}
