// network.cu - building the network: random wiring (fixed mean in-degree), synapse lists
#include "brain.cuh"
#include <functional>

// k distinct integers from [lo, hi) excluding `skip`, sorted ascending
static void sample_distinct(std::mt19937& gen, int lo, int hi, int skip, int k, std::vector<int>& out) {
    out.clear();
    const int avail = hi - lo - ((skip >= lo && skip < hi) ? 1 : 0);
    k = std::min(k, avail);
    std::uniform_int_distribution<int> U(lo, hi - 1);
    while ((int)out.size() < k) {
        const int need = k - (int)out.size();
        for (int i = 0; i < need; ++i) {
            const int v = U(gen);
            if (v != skip) out.push_back(v);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }
}


// =============================================================================
//  Sound-mode anatomy: positions on the sphere, distance-dependent wiring, auditory pathway
// =============================================================================
namespace {
struct V3 { float x, y, z; };
inline V3 v3(float x, float y, float z) { return V3{ x, y, z }; }
inline float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 norm3(V3 a) { const float l = std::sqrt(dot3(a, a)); return v3(a.x / l, a.y / l, a.z / l); }
inline V3 cross3(V3 a, V3 b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
inline V3 sph(float lon, float lat) {             // lon 0 = front (+z), +lon = right (+x), lat + = up
    return v3(std::cos(lat) * std::sin(lon), std::sin(lat), std::cos(lat) * std::cos(lon));
}
inline float ang(V3 a, V3 b) { return std::acos(std::max(-1.f, std::min(1.f, dot3(a, b)))); }

// cube-map grid on the sphere: cell index of a unit vector
inline int cube_cell(V3 p, int G) {
    const float ax = std::fabs(p.x), ay = std::fabs(p.y), az = std::fabs(p.z);
    int f; float u, v;
    if (ax >= ay && ax >= az) { f = p.x > 0 ? 0 : 1; u = p.y / ax; v = p.z / ax; }
    else if (ay >= az)        { f = p.y > 0 ? 2 : 3; u = p.x / ay; v = p.z / ay; }
    else                      { f = p.z > 0 ? 4 : 5; u = p.x / az; v = p.y / az; }
    const int iu = std::min(G - 1, std::max(0, (int)((u + 1.f) * 0.5f * G)));
    const int iv = std::min(G - 1, std::max(0, (int)((v + 1.f) * 0.5f * G)));
    return (f * G + iu) * G + iv;
}
}  // namespace

static void build_anatomy(Network& net, const Anatomy& A, const Params& p, int N, int N_E,
                          int d_min, int d_max, double p_from_E, double p_from_I, std::mt19937& gen) {
    std::uniform_real_distribution<float> U(0.f, 1.f);
    std::normal_distribution<float> G01(0.f, 1.f);
    const float sigma2 = std::log(2.f);
    std::lognormal_distribution<float> LN(-sigma2 / 2.f, std::sqrt(sigma2));
    std::uniform_int_distribution<int> Br(0, B - 1);
    const int NB = EAR_BANDS, TP = EAR_PER;
    const float PI = 3.14159265f;

    // ---- positions and regions ----
    std::vector<V3> pos(N);
    std::vector<uint8_t> reg(N, REG_CORTEX);
    std::vector<int> band(N, -1);                  // nerve / A1: frequency band; mouth: tone
    const float a1_lat = -0.15f, a1_lon = 1.45f, a1_r = 0.38f;
    const V3 a1c[2] = { sph(-a1_lon, a1_lat), sph(a1_lon, a1_lat) };
    for (int n = 0; n < N; ++n) {
        if (n < A.ear_n) {                         // auditory nerve: drawn as strips behind each A1
            reg[n] = REG_NERVE;
            const int b = n / TP, k = n % TP, side = k & 1, j = k >> 1;
            band[n] = b;
            const float lat = a1_lat - a1_r + 2.f * a1_r * (b + 0.5f) / NB;
            const float lon = (side ? 1.f : -1.f) * (a1_lon + 0.55f + 0.18f * ((j + 0.5f) / (TP / 2) - 0.5f));
            pos[n] = sph(lon, lat);
        } else if (n >= A.mouth0 && n < A.mouth0 + A.mouth_n) {   // mouth: frontal patch
            reg[n] = REG_MOUTH;
            const int t = (n - A.mouth0) / MOUTH_PER, k = (n - A.mouth0) % MOUTH_PER;
            band[n] = t;
            const float lon = 0.9f * ((t + 0.5f) / MOUTH_TONES - 0.5f);
            const float lat = 0.30f + 0.16f * (k + 0.5f) / MOUTH_PER;
            pos[n] = sph(lon, lat);
        } else {                                   // cortex: uniform on the sphere
            V3 d;
            for (;;) { d = v3(G01(gen), G01(gen), G01(gen)); if (dot3(d, d) > 1e-6f) break; }
            pos[n] = norm3(d);
            for (int s = 0; s < 2; ++s) {
                const float a = ang(pos[n], a1c[s]);
                if (a < a1_r) {                    // inside A1: pitch map along latitude
                    reg[n] = REG_A1;
                    const float lat = std::asin(std::max(-1.f, std::min(1.f, pos[n].y)));
                    const float u = (lat - (a1_lat - a1_r)) / (2.f * a1_r);
                    band[n] = std::min(NB - 1, std::max(0, (int)(u * NB)));
                }
            }
        }
    }

    // ---- spatial index of cortical neurons (nerve excluded), E and I separately ----
    const int Gc = std::max(4, (int)std::sqrt((double)N / (6.0 * 24.0)));
    std::vector<std::vector<int>> cellE(6 * Gc * Gc), cellI(6 * Gc * Gc);
    std::vector<int> allE, allI;
    for (int n = 0; n < N; ++n) {
        if (reg[n] == REG_NERVE) continue;
        (n < N_E ? cellE : cellI)[cube_cell(pos[n], Gc)].push_back(n);
        (n < N_E ? allE : allI).push_back(n);
    }
    std::vector<std::vector<int>> nerve_by_band(NB), a1E_by_band(NB);
    for (int n = 0; n < N; ++n) {
        if (reg[n] == REG_NERVE) nerve_by_band[band[n]].push_back(n);
        if (reg[n] == REG_A1 && n < N_E) a1E_by_band[band[n]].push_back(n);
    }

    // one presynaptic candidate near neuron i (or anywhere, with probability p_long)
    auto near_pick = [&](int i, bool fromE) -> int {
        const auto& all = fromE ? allE : allI;
        if (all.empty()) return -1;
        if (U(gen) >= A.p_long) {
            const V3 c = pos[i];
            const V3 ax = std::fabs(c.x) < 0.9f ? v3(1, 0, 0) : v3(0, 1, 0);
            const V3 e1 = norm3(cross3(c, ax)), e2 = cross3(c, e1);
            for (int tries = 0; tries < 16; ++tries) {
                const float d = std::min(PI, -A.lambda * std::log(std::max(1e-7f, U(gen))));
                const float th = 2.f * PI * U(gen);
                const float cd = std::cos(d), sd = std::sin(d), ct = std::cos(th), st = std::sin(th);
                const V3 q = v3(c.x * cd + (e1.x * ct + e2.x * st) * sd,
                                c.y * cd + (e1.y * ct + e2.y * st) * sd,
                                c.z * cd + (e1.z * ct + e2.z * st) * sd);
                const auto& cell = (fromE ? cellE : cellI)[cube_cell(q, Gc)];
                if (!cell.empty()) return cell[(size_t)(U(gen) * cell.size()) % cell.size()];
            }
        }
        return all[(size_t)(U(gen) * all.size()) % all.size()];
    };
    // k distinct inputs from a generator, excluding i (topping up after removing duplicates)
    std::vector<int> picks, extra;
    auto fill = [&](int i, int k, const std::function<int()>& draw, std::vector<int>& out) {
        out.clear();
        for (int round = 0; round < 50 && (int)out.size() < k; ++round) {
            const int need = k - (int)out.size();
            for (int t = 0; t < need; ++t) { const int v = draw(); if (v >= 0 && v != i) out.push_back(v); }
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
        }
        if ((int)out.size() > k) out.resize(k);
    };
    auto delay_of = [&](int pre, int post) {       // 1 ms nearby, up to 3 ms across the head
        const float d = ang(pos[pre], pos[post]);
        int dl = d_min + (int)std::lround((d_max - d_min) * std::min(1.f, d / 1.6f));
        dl += (int)(U(gen) * 3.f) - 1;              // a little jitter
        return std::max(d_min, std::min(d_max, dl));
    };

    auto& in_ptr = net.in_ptr; auto& in_eend = net.in_eend; auto& in_pre = net.in_pre;
    auto& in_w = net.in_w; auto& in_meta = net.in_meta;
    const int N_I = N - N_E;
    for (int i = 0; i < N; ++i) {
        const bool postE = i < N_E;
        for (int src = 0; src < 2; ++src) {
            const bool fromE = src == 0;
            const int pool = (fromE ? N_E : N_I) - ((postE == fromE) ? 1 : 0);
            std::binomial_distribution<int> K(pool, fromE ? p_from_E : p_from_I);
            int k = K(gen);
            picks.clear();
            if (fromE && reg[i] == REG_NERVE) {
                k = 0;                             // auditory nerve: no excitatory input from the cortex
            } else if (fromE && reg[i] == REG_A1) {
                const int kn = std::min(A.k_nerve, k);   // pitch-matched nerve fibres
                fill(i, kn, [&]() {
                    const int b = std::min(NB - 1, std::max(0, band[i] + (int)std::lround(1.2f * G01(gen))));
                    const auto& v = nerve_by_band[b];
                    return v.empty() ? -1 : v[(size_t)(U(gen) * v.size()) % v.size()];
                }, extra);
                fill(i, k - (int)extra.size(), [&]() { return near_pick(i, true); }, picks);
                picks.insert(picks.end(), extra.begin(), extra.end());
            } else if (fromE && reg[i] == REG_MOUTH) {
                const int ka = std::min(A.k_a1_mouth, k); // A1 neurons tuned to the matching pitch
                const int b0 = (int)((band[i] + 0.5f) * NB / MOUTH_TONES);
                fill(i, ka, [&]() {
                    const int b = std::min(NB - 1, std::max(0, b0 + (int)std::lround(1.5f * G01(gen))));
                    const auto& v = a1E_by_band[b];
                    return v.empty() ? -1 : v[(size_t)(U(gen) * v.size()) % v.size()];
                }, extra);
                fill(i, k - (int)extra.size(), [&]() { return near_pick(i, true); }, picks);
                picks.insert(picks.end(), extra.begin(), extra.end());
            } else {
                fill(i, k, [&]() { return near_pick(i, fromE); }, picks);
            }
            std::sort(picks.begin(), picks.end());
            picks.erase(std::unique(picks.begin(), picks.end()), picks.end());
            for (int j : picks) {
                float wv = LN(gen) * (fromE ? p.g_exc : p.g_inh);
                if (fromE && postE) wv = std::min(wv, p.w_max);
                in_pre.push_back(j);
                in_w.push_back(wv);
                in_meta.push_back((uint16_t)(delay_of(j, i) | (Br(gen) << 6)));
            }
            if (fromE) in_eend[i] = (int)in_pre.size();
        }
        in_ptr[i + 1] = (int)in_pre.size();
    }

    // ---- layout for the viewer: shell depth for looks; colour types 3 nerve, 4 A1, 5 mouth ----
    net.xyzt.resize((size_t)N * 4);
    for (int n = 0; n < N; ++n) {
        const float R = 1.05f * (reg[n] == REG_CORTEX || reg[n] == REG_A1 ? 0.80f + 0.20f * std::cbrt(U(gen)) : 1.0f);
        float* o = &net.xyzt[(size_t)n * 4];
        o[0] = pos[n].x * R; o[1] = pos[n].y * R; o[2] = pos[n].z * R;
        o[3] = reg[n] == REG_NERVE ? 3.f : reg[n] == REG_A1 ? 4.f : reg[n] == REG_MOUTH ? 5.f : (n < N_E ? 1.f : 2.f);
    }
    net.region = reg;
    net.band.assign(band.begin(), band.end());
    int nA1 = 0; for (int n = 0; n < N; ++n) nA1 += reg[n] == REG_A1;
    std::printf("  anatomy: auditory nerve %d, A1 %d (both sides), mouth %d; local wiring (lambda %.2f rad, %.0f%% long-range)\n",
                A.ear_n, nA1, A.mouth_n, A.lambda, 100.f * A.p_long);
}


// Outgoing lists and counts from the incoming lists (also used when a brain is loaded from a file)
void finish_network(Network& net, int N, int N_E, int N_A) {
    auto& in_ptr = net.in_ptr; auto& in_eend = net.in_eend; auto& in_pre = net.in_pre;
    auto& in_meta = net.in_meta;
    const size_t nnz = in_pre.size();
    net.nnz = nnz;

    // outgoing CSR: for each presynaptic neuron, its targets and the synapse's slot in their lists
    auto& out_ptr = net.out_ptr; auto& out_post = net.out_post; auto& out_q = net.out_q;
    out_ptr.assign(N + 1, 0); out_post.assign(nnz, 0); out_q.assign(nnz, 0);
    net.out_meta.assign(nnz, 0);
    for (size_t q = 0; q < nnz; ++q) out_ptr[in_pre[q] + 1]++;
    for (int j = 0; j < N; ++j) out_ptr[j + 1] += out_ptr[j];
    {
        std::vector<int> cur(out_ptr.begin(), out_ptr.end() - 1);
        for (int i = 0; i < N; ++i)
            for (int q = in_ptr[i]; q < in_ptr[i + 1]; ++q) {
                const int pos = cur[in_pre[q]]++;
                out_post[pos] = i;
                out_q[pos]    = q;
                net.out_meta[pos] = (uint16_t)(in_meta[q] | ((in_pre[q] < N_E ? 1u : 0u) << 8));
            }
    }

    long long& cntA = net.cntA; long long& cntO = net.cntO;
    cntA = cntO = 0;
    for (int i = 0; i < N_E; ++i)
        for (int q = in_ptr[i]; q < in_eend[i]; ++q)
            ((in_pre[q] < N_A && i < N_A) ? cntA : cntO)++;

}

Network build_network(const Params& p, int N, int N_E, int N_A, int d_min, int d_max,
                      double p_from_E, double p_from_I, const Anatomy* anat) {
    const int N_I = N - N_E;
    std::printf("Building network: %d neurons (%d E, %d I), ~%.0f local inputs each...\n",
                N, N_E, N_I, p_from_E * (N_E - 1) + p_from_I * (N_I - 1));
    const auto tb = std::chrono::high_resolution_clock::now();
    std::mt19937 gen(2);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    const float sigma2 = std::log(2.f);
    std::lognormal_distribution<float> LN(-sigma2 / 2.f, std::sqrt(sigma2));
    std::uniform_int_distribution<int> Dist(d_min, d_max), Br(0, B - 1);

    Network net;
    auto& in_ptr = net.in_ptr; auto& in_eend = net.in_eend; auto& in_pre = net.in_pre;
    auto& in_w = net.in_w; auto& in_meta = net.in_meta;
    in_ptr.assign(N + 1, 0); in_eend.assign(N, 0);
    const size_t nnz_guess = (size_t)N * (size_t)(p_from_E * N_E + p_from_I * N_I + 8);
    in_pre.reserve(nnz_guess); in_w.reserve(nnz_guess); in_meta.reserve(nnz_guess);
    std::vector<int> picks;
    if (anat && anat->on) {
        build_anatomy(net, *anat, p, N, N_E, d_min, d_max, p_from_E, p_from_I, gen);
    } else
    for (int i = 0; i < N; ++i) {
        const bool postE = i < N_E;
        for (int src = 0; src < 2; ++src) {               // 0: E inputs, 1: I inputs
            const bool fromE = src == 0;
            const int  lo = fromE ? 0 : N_E, hi = fromE ? N_E : N;
            const int  pool = (hi - lo) - ((i >= lo && i < hi) ? 1 : 0);
            std::binomial_distribution<int> K(pool, fromE ? p_from_E : p_from_I);
            sample_distinct(gen, lo, hi, i, K(gen), picks);
            for (int j : picks) {
                float wv = LN(gen) * (fromE ? p.g_exc : p.g_inh);
                if (fromE && postE) wv = std::min(wv, p.w_max);   // plastic: STDP cap from the start
                in_pre.push_back(j);
                in_w.push_back(wv);
                in_meta.push_back((uint16_t)(Dist(gen) | (Br(gen) << 6)));
            }
            if (fromE) in_eend[i] = (int)in_pre.size();
        }
        in_ptr[i + 1] = (int)in_pre.size();
    }
    finish_network(net, N, N_E, N_A);
    const size_t nnz = net.nnz;
    const long long cntA = net.cntA, cntO = net.cntO;

    auto& Vs0 = net.Vs0; auto& Vb0 = net.Vb0;
    Vs0.assign(N, 0.f); Vb0.assign((size_t)N * B, 0.f);
    for (int n = 0; n < N; ++n) {
        Vs0[n] = p.EL + 10.f * U(gen);
        for (int k = 0; k < B; ++k) Vb0[(size_t)n * B + k] = Vs0[n];
    }
    std::printf("  %zu synapses (%.1f per neuron, %.0f plastic per E neuron), built in %.1f s\n",
                nnz, (double)nnz / N, (double)(cntA + cntO) / N_E,
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tb).count());
    return net;
}
