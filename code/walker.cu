// walker.cu - walking creatures: each brain drives a two-legged body and learns from dopamine.
//
//   sensors -> brain:  joint angles and speeds, torso tilt and spin, forward speed, foot contact,
//                      each as a pair of neuron groups (one for +, one for -), firing in proportion
//   brain -> muscles:  for each joint a flex group and an extend group; the difference between
//                      their firing rates sets the joint motor's speed (like opposing muscles)
//   dopamine:          continuously, in proportion to forward speed (backwards counts double, as
//                      punishment); tipping past the death tilt (75 deg) or
//                      lying down kills it: punished, its distance forfeited, and it is born again
//   selection (opt.):  every few lives, the worst walker's brain becomes a copy of the best one's
#include "brain.cuh"
#include "walkphys.h"
#include <cmath>
#include <deque>
#include <cstdio>
#include <string>
#include <random>
#include <algorithm>

// ---------------------------------------------------------------------------
//  Where the senses and muscles live inside one brain (local excitatory indices)
// ---------------------------------------------------------------------------
WalkLayout walk_layout(int n, bool modular) {
    WalkLayout L;
    L.modular = modular;
    const int nE = (int)std::lround(0.8 * n), nI = n - nE;
    if (!modular) {
        L.modE[0] = 0; L.modE[1] = L.modE[2] = L.modE[3] = nE;
        L.modI[0] = 0; L.modI[1] = L.modI[2] = L.modI[3] = nI;
        for (int g = 0; g < WALK_SENSE_GROUPS; ++g) L.sense_start[g] = g * WALK_SENSE_PER;
        const int motor0 = nE - WALK_MOTOR_GROUPS * WALK_MOTOR_PER;
        for (int m = 0; m < WALK_MOTOR_GROUPS; ++m) L.motor_start[m] = motor0 + m * WALK_MOTOR_PER;
        return L;
    }
    // modules: torso 30 %, back leg and front leg 35 % each (of the E cells, and of the I cells)
    const int tE = (int)std::lround(0.30 * nE), lE = (nE - tE) / 2;
    const int tI = (int)std::lround(0.30 * nI), lI = (nI - tI) / 2;
    L.modE[0] = 0; L.modE[1] = tE; L.modE[2] = tE + lE; L.modE[3] = nE;
    L.modI[0] = 0; L.modI[1] = tI; L.modI[2] = tI + lI; L.modI[3] = nI;
    int next[3] = { L.modE[0], L.modE[1], L.modE[2] };             // next free sensory cell per module
    auto place = [&](int g, int mod) { L.sense_start[g] = next[mod]; next[mod] += WALK_SENSE_PER; };
    for (int j = 0; j < WALK_JOINTS; ++j) {                          // a leg feels only its own joints
        const int mod = 1 + j / 2;
        place(2 * j, mod); place(2 * j + 1, mod);                    // angle + / -
        place(8 + 2 * j, mod); place(9 + 2 * j, mod);                // speed + / -
    }
    for (int g = 16; g < 22; ++g) place(g, 0);                       // torso: tilt, spin, forward speed
    place(22, 1); place(23, 2);                                      // each leg: its own foot
    for (int m = 0; m < WALK_MOTOR_GROUPS; ++m) {                    // a leg moves only its own joints
        const int j = m / 2, mod = 1 + j / 2, k = m % 4;             // k: 0-3 within the leg
        L.motor_start[m] = L.modE[mod + 1] - (4 - k) * WALK_MOTOR_PER;
    }
    return L;
}

// One octopus brain: three modules, each wired mostly inside itself. Same number of inputs per
// cell as the blob brain (about 80 excitatory + 80 inhibitory), only where they come from differs:
//   excitatory: 85 % from the cell's own module, 15 % from the other two (the conversation)
//   inhibitory: local, except that a quarter of a leg module's inhibition comes from the other leg
//               (crossed inhibition, like the spinal cord's commissural interneurons: one leg
//               pushing holds the other back, which favours alternating legs)
//   delays: 1-2 ms inside a module, 2-3 ms between modules
static Network build_octopus_brain(const Params& p, int n, int d_min, int d_max, float kE, float kI) {
    const WalkLayout W = walk_layout(n, true);
    const int nE = W.modE[3], nI = n - nE;
    std::mt19937 gen(2);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    const float sigma2 = std::log(2.f);
    std::lognormal_distribution<float> LN(-sigma2 / 2.f, std::sqrt(sigma2));
    const int d_mid = (d_min + d_max) / 2;
    std::uniform_int_distribution<int> Dnear(d_min, d_mid), Dfar(d_mid, d_max), Br(0, B - 1);
    const float OWN_E = 0.85f, CROSS_I = 0.25f;

    auto mod_of = [&](int i) {                         // local index -> module 0..2
        if (i < nE) return i < W.modE[1] ? 0 : i < W.modE[2] ? 1 : 2;
        const int k = i - nE; return k < W.modI[1] ? 0 : k < W.modI[2] ? 1 : 2;
    };
    // k distinct cells from a set of [lo, hi) ranges (local indices), never `self`
    std::vector<int> picks;
    auto pick = [&](const std::vector<std::pair<int, int>>& R, int k, int self) {
        int total = 0; for (auto& r : R) total += r.second - r.first;
        const int has_self = [&] { for (auto& r : R) if (self >= r.first && self < r.second) return 1; return 0; }();
        k = std::min(k, total - has_self);
        const size_t base = picks.size();
        std::vector<int> got;
        while ((int)got.size() < k) {
            int v = (int)(U(gen) * total); if (v >= total) v = total - 1;
            for (auto& r : R) { const int len = r.second - r.first; if (v < len) { v += r.first; break; } v -= len; }
            if (v == self) continue;
            got.push_back(v);
            std::sort(got.begin(), got.end());
            got.erase(std::unique(got.begin(), got.end()), got.end());
        }
        picks.insert(picks.begin() + base, got.begin(), got.end());
    };
    auto Erange = [&](int m) { return std::make_pair(W.modE[m], W.modE[m + 1]); };
    auto Irange = [&](int m) { return std::make_pair(nE + W.modI[m], nE + W.modI[m + 1]); };

    Network net;
    net.in_ptr.assign(n + 1, 0); net.in_eend.assign(n, 0);
    long long between = 0, cross_inh = 0;
    for (int i = 0; i < n; ++i) {
        const int m = mod_of(i);
        for (int src = 0; src < 2; ++src) {
            const bool fromE = src == 0;
            const int pool = (fromE ? nE : nI) - 1;
            std::binomial_distribution<int> K(pool, std::min(1.0, (double)(fromE ? kE : kI) / pool));
            const int k = K(gen);
            picks.clear();
            if (fromE) {
                std::binomial_distribution<int> Own(k, OWN_E);
                const int ko = Own(gen);
                pick({ Erange(m) }, ko, i);
                std::vector<std::pair<int, int>> others;
                for (int o = 0; o < 3; ++o) if (o != m) others.push_back(Erange(o));
                pick(others, k - ko, i);
            } else if (m == 0) {
                pick({ Irange(0) }, k, i);
            } else {
                std::binomial_distribution<int> Cr(k, CROSS_I);
                const int kc = Cr(gen);
                pick({ Irange(m) }, k - kc, i);
                pick({ Irange(m == 1 ? 2 : 1) }, kc, i);
            }
            std::sort(picks.begin(), picks.end());
            picks.erase(std::unique(picks.begin(), picks.end()), picks.end());
            for (int j : picks) {
                float wv = LN(gen) * (fromE ? p.g_exc : p.g_inh);
                if (fromE && i < nE) wv = std::min(wv, p.w_max);
                const bool far = mod_of(j) != m;
                if (far) { if (fromE) between++; else cross_inh++; }
                net.in_pre.push_back(j);
                net.in_w.push_back(wv);
                net.in_meta.push_back((uint16_t)((far ? Dfar(gen) : Dnear(gen)) | (Br(gen) << 6)));
            }
            if (fromE) net.in_eend[i] = (int)net.in_pre.size();
        }
        net.in_ptr[i + 1] = (int)net.in_pre.size();
    }
    net.nnz = net.in_pre.size();
    net.Vs0.assign(n, 0.f); net.Vb0.assign((size_t)n * B, 0.f);
    for (int c = 0; c < n; ++c) {
        net.Vs0[c] = p.EL + 10.f * U(gen);
        for (int k = 0; k < B; ++k) net.Vb0[(size_t)c * B + k] = net.Vs0[c];
    }
    std::printf("  octopus brain: torso %d, back leg %d, front leg %d neurons; %zu synapses "
                "(%.0f%% between modules, %lld crossed-inhibition links between the legs)\n",
                W.modE[1] + W.modI[1], (W.modE[2] - W.modE[1]) + (W.modI[2] - W.modI[1]),
                (W.modE[3] - W.modE[2]) + (W.modI[3] - W.modI[2]), net.nnz,
                100.0 * (between + cross_inh) / std::max<size_t>(1, net.nnz), cross_inh);
    return net;
}

// ---------------------------------------------------------------------------
//  Spinal rhythm generators (octopus brains, optional): each joint's flex and extend motor groups
//  are wired as a HALF-CENTRE OSCILLATOR, the classic spinal walking circuit (Graham Brown, 1911):
//    - each group excites itself (a burst sustains itself)
//    - each group drives its own pool of inhibitory cells, which silences the opposite group
//    - the E cells' own adaptation (b = 60 pA, tau_w = 200 ms) tires the active group out, the
//      other group escapes, and the cycle repeats: flex, extend, flex, extend
//  Knee follows hip: the hip group that swings the foot forward (swing) excites knee FLEX (lift the
//  foot); the hip group that pushes back (stance) excites knee EXTEND (straight leg to push on).
//  A steady "brainstem drive" into the motor groups sets how hard the circuits run (run_walk).
//  Tested on the CPU bench before going in: rhythm ~3 Hz per joint, knee locked to hip.
// ---------------------------------------------------------------------------
namespace {
struct CpgSyn { int post, pre; float w; int d, br; };
constexpr float CPG_W_SELF = 1.0f, CPG_W_EI = 0.8f, CPG_W_IE = 1.0f, CPG_W_HK = 0.5f;   // nS
constexpr int   CPG_K_SELF = 20, CPG_K_EI = 20, CPG_POOL = 15, CPG_K_HK = 10;

std::vector<CpgSyn> cpg_wiring(int n, int dly) {
    const WalkLayout L = walk_layout(n, true);
    const int nE = L.modE[3];
    std::mt19937 g(11);
    std::uniform_int_distribution<int> Br(0, B - 1);
    std::vector<CpgSyn> ex;
    auto grp  = [&](int m) { return L.motor_start[m]; };            // m = 2*joint + (0 flex, 1 extend)
    auto pool = [&](int m) { const int mod = 1 + (m / 2) / 2; return nE + L.modI[mod] + (m % 4) * CPG_POOL; };
    auto pick = [&](int lo, int cnt, int k, int self) {
        std::vector<int> v; std::uniform_int_distribution<int> U(0, cnt - 1);
        while ((int)v.size() < k) { const int x = lo + U(g); if (x != self && std::find(v.begin(), v.end(), x) == v.end()) v.push_back(x); }
        return v;
    };
    for (int m = 0; m < WALK_MOTOR_GROUPS; ++m) {
        const int opp = m ^ 1;
        for (int i = 0; i < WALK_MOTOR_PER; ++i) {
            const int post = grp(m) + i;
            for (int pre : pick(grp(m), WALK_MOTOR_PER, CPG_K_SELF, post)) ex.push_back({ post, pre, CPG_W_SELF, dly, Br(g) });
            for (int pi = 0; pi < CPG_POOL; ++pi) ex.push_back({ post, pool(opp) + pi, CPG_W_IE, dly, 0 });
        }
        for (int pi = 0; pi < CPG_POOL; ++pi)
            for (int pre : pick(grp(m), WALK_MOTOR_PER, CPG_K_EI, -1)) ex.push_back({ pool(m) + pi, pre, CPG_W_EI, dly, 0 });
    }
    for (int leg = 0; leg < 2; ++leg)                                // joints: 0/1 back hip/knee, 2/3 front
        for (int ph = 0; ph < 2; ++ph) {
            const int hip = 2 * (2 * leg) + ph, knee = 2 * (2 * leg + 1) + (1 - ph);
            for (int i = 0; i < WALK_MOTOR_PER; ++i)
                for (int pre : pick(grp(hip), WALK_MOTOR_PER, CPG_K_HK, -1)) ex.push_back({ grp(knee) + i, pre, CPG_W_HK, dly, Br(g) });
        }
    return ex;
}

// merge extra synapses into a one-brain network, keeping each neuron's E inputs first
void add_synapses(Network& a, int n, int nE, const std::vector<CpgSyn>& ex) {
    std::vector<std::vector<CpgSyn>> per(n);
    for (auto& s : ex) per[s.post].push_back(s);
    Network b = a;
    b.in_ptr.assign(1, 0); b.in_eend.assign(n, 0); b.in_pre.clear(); b.in_w.clear(); b.in_meta.clear();
    for (int i = 0; i < n; ++i) {
        for (int pass = 0; pass < 2; ++pass) {
            const int q0 = pass == 0 ? a.in_ptr[i] : a.in_eend[i], q1 = pass == 0 ? a.in_eend[i] : a.in_ptr[i + 1];
            for (int q = q0; q < q1; ++q) { b.in_pre.push_back(a.in_pre[q]); b.in_w.push_back(a.in_w[q]); b.in_meta.push_back(a.in_meta[q]); }
            for (auto& s : per[i]) if ((s.pre < nE) == (pass == 0)) {
                b.in_pre.push_back(s.pre); b.in_w.push_back(s.w); b.in_meta.push_back((uint16_t)(s.d | (s.br << 6)));
            }
            if (pass == 0) b.in_eend[i] = (int)b.in_pre.size();
        }
        b.in_ptr.push_back((int)b.in_pre.size());
    }
    b.nnz = b.in_pre.size();
    a = std::move(b);
}
}  // namespace

// ---------------------------------------------------------------------------
//  Network: K identical copies of one brain, no connections between them.
//  Global layout keeps all excitatory cells first: brain b's E cells are [b*nE, (b+1)*nE),
//  its I cells [K*nE + b*nI, K*nE + (b+1)*nI).
// ---------------------------------------------------------------------------
Network build_walk_network(const Params& p, int K, int n, int d_min, int d_max, bool modular, float kE, float kI, bool cpg) {
    const int nE = (int)std::lround(0.8 * n), nI = n - nE;
    std::printf("Building %d identical %s brains of %d neurons (%d E, %d I)...\n",
                K, modular ? "octopus" : "blob", n, nE, nI);
    Network one = modular ? build_octopus_brain(p, n, d_min, d_max, kE, kI)
                          : build_network(p, n, nE, nE / 4, d_min, d_max,
                                          std::min(1.0, (double)kE / nE), std::min(1.0, (double)kI / nI));
    if (modular && cpg) {
        const auto ex = cpg_wiring(n, d_min);
        add_synapses(one, n, nE, ex);
        std::printf("  spinal rhythm generators: a half-centre oscillator per joint (%zu extra synapses per brain)\n", ex.size());
    }
    const int N = K * n, N_E = K * nE;
    auto G = [&](int b, int i) { return i < nE ? b * nE + i : N_E + b * nI + (i - nE); };
    Network net;
    net.in_ptr.assign(1, 0);
    net.in_eend.assign(N, 0);
    for (int g = 0; g < N; ++g) {                       // incoming lists in global post order
        int b, i;
        if (g < N_E) { b = g / nE; i = g % nE; }
        else { b = (g - N_E) / nI; i = nE + (g - N_E) % nI; }
        for (int q = one.in_ptr[i]; q < one.in_ptr[i + 1]; ++q) {
            net.in_pre.push_back(G(b, one.in_pre[q]));
            net.in_w.push_back(one.in_w[q]);
            net.in_meta.push_back(one.in_meta[q]);
        }
        net.in_eend[g] = net.in_ptr.back() + (one.in_eend[i] - one.in_ptr[i]);
        net.in_ptr.push_back((int)net.in_pre.size());
    }
    net.nnz = net.in_pre.size();
    net.Vs0.resize(N); net.Vb0.resize((size_t)N * B);
    for (int b = 0; b < K; ++b)
        for (int i = 0; i < n; ++i) {
            const int g = G(b, i);
            net.Vs0[g] = one.Vs0[i];
            for (int k = 0; k < B; ++k) net.Vb0[(size_t)g * B + k] = one.Vb0[(size_t)i * B + k];
        }
    finish_network(net, N, N_E, N_E / 4);
    return net;
}

// Dopamine for several brains at once: brain b = n / nE gets amounts[b]
// Mutation: every synapse onto brain `dst` is scaled by exp(sigma * N(0,1)) (multiplicative, so weights
// stay positive and a 0 stays 0), then clamped like the network's own weights (E->E at most w_max).
// Random numbers: counter-based Philox from (synapse, seed), like the simulation's.
__global__ void walk_mutate_kernel(const int* __restrict__ in_ptr, const int* __restrict__ in_eend,
                                   float* __restrict__ in_w, int N, int N_E, int nE, int nI, int dst,
                                   float sigma, unsigned seed, float w_max) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= N) return;
    const int b = g < N_E ? g / nE : (g - N_E) / nI;
    if (b != dst) return;
    const uint2 key = make_uint2(0x6D2B79F5u, seed);
    for (int q = in_ptr[g]; q < in_ptr[g + 1]; ++q) {
        const float w = in_w[q];
        if (w == 0.0f) continue;
        const uint4 r = curand_Philox4x32_10(make_uint4((unsigned)q, seed, 0xE701u, 0u), key);
        const float u1 = ((r.x >> 8) + 1.0f) * (1.0f / 16777217.0f), u2 = (r.y >> 8) * (1.0f / 16777216.0f);
        const float z  = sqrtf(-2.0f * __logf(u1)) * __cosf(6.2831853f * u2);   // Box-Muller: N(0,1)
        float nw = w * __expf(sigma * z);
        if (g < N_E && q < in_eend[g]) nw = fminf(nw, w_max);           // plastic E->E: same cap as learning
        in_w[q] = nw;
    }
}

__global__ void walk_reward_kernel(const int* __restrict__ in_ptr, const int* __restrict__ in_eend,
                                   float* __restrict__ in_w, const float* __restrict__ elig,
                                   const int* __restrict__ elig_t, int N_E, int nE, int now, float k_elig,
                                   const float* __restrict__ amounts, float w_max) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N_E) return;
    const float a = amounts[n / nE];
    if (a == 0.0f) return;
    for (int q = in_ptr[n]; q < in_eend[n]; ++q) {
        const float e = elig[q] * __expf(-(float)(now - elig_t[q]) * k_elig);
        if (e != 0.0f) in_w[q] = fminf(fmaxf(in_w[q] + a * e, 0.0f), w_max);
    }
}

#if defined(BRAIN_VIZ) && defined(BRAIN_VIZ_UI)
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>                // before GLFW's OpenGL header (APIENTRY)
#endif
#include <GLFW/glfw3.h>             // OpenGL 1.1 header: all this window calls directly
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#ifdef _WIN32
#pragma comment(lib, "implot.lib")
#endif

namespace {
const ImU32 kColors[8] = { IM_COL32(255, 160, 40, 255), IM_COL32(60, 200, 255, 255), IM_COL32(120, 230, 90, 255),
                           IM_COL32(240, 90, 200, 255), IM_COL32(250, 230, 70, 255), IM_COL32(150, 130, 255, 255),
                           IM_COL32(255, 100, 100, 255), IM_COL32(200, 200, 200, 255) };
ImU32 with_alpha(ImU32 c, float a) { return (c & 0x00FFFFFFu) | ((ImU32)(a * 255.f) << 24); }

struct Life {                       // one creature's current life and record
    float start_x = 0.f, t = 0.f, fall_timer = -1.f, dop_rate = 0.f;
    float expect = 0.f;             // reward per second this brain has come to expect (running average)
    int   lives = 0, deaths = 0, since_selection = 0;
    float last = 0.f, best = -1e9f;
    std::deque<float> recent;       // distances of the last few lives
    std::vector<float> history;     // distance of every life (for the graph)
    float cmd[WALK_JOINTS] = { 0, 0, 0, 0 };
};
}  // namespace

int run_walk(DevState& s, Params& p, int blocks, int threads, const WalkCtx& ctx) {
    const int K = ctx.brains, n = ctx.n, nE = (int)std::lround(0.8 * n), nI = n - nE;
    const int N = p.N, N_E = p.N_E;
    const WalkLayout lay = walk_layout(n, ctx.modular);
    WalkWorld world(K);

    // ---- window ----
    if (!glfwInit()) { std::fprintf(stderr, "GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* win = glfwCreateWindow(1600, 900, "Brain - walking creatures", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "Could not open a window\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // ---- buffers ----
    std::vector<int> h_in_ptr(N + 1);
    CUDA_CHECK(cudaMemcpy(h_in_ptr.data(), s.in_ptr, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<float> sense((size_t)N_E, 0.f), amounts(K, 0.f);
    std::vector<unsigned> spk((size_t)N_E);
    float* d_amt = nullptr; CUDA_CHECK(cudaMalloc(&d_amt, K * sizeof(float)));
    std::vector<Life> life(K);

    // ---- settings (panel) ----
    float speed = 1.0f;                 // simulated seconds per real second
    bool  fast = true;                  // run as fast as possible (draw less often)
    float reward_gain = 10.f, back_mult = 2.f, fall_penalty = 5.f, life_len = 30.f;
    float death_tilt = 45.f;            // degrees: torso tipped further than this = dead
    bool  kill_back = true;             // walking backwards past kill_back_m behind the start = dead
    float kill_back_m = 1.0f;
    bool  forfeit = true;               // a death scores no forward distance (selection, graph)
    float sensor_gain = 0.3f, motor_norm = ctx.cpg ? 150.f : 10.f, elig_tau = 1.0f;
    float cpg_drive = 0.2f;             // steady "brainstem" drive into the motor groups (rhythm generators)
    float reflex = 3.f;                 // stretch reflex: pull of each joint back to its resting angle
    float lean = 0.0f;                  // resting hip angle (rad): posture leaning into the walk
    float alive_reward = 1.f;           // dopamine per second for staying alive and upright
    bool  surprise = true;              // dopamine = reward minus the reward the brain expects
    float expect_tau = 10.f;            // seconds over which the expectation is averaged
    bool  selection = true; int sel_lives = 3;
    float replace_frac = 0.5f;          // share of the population replaced each generation
    float mutation = 0.05f;             // mutation strength: typical relative change per synapse
    int   generation = 0;
    int   follow = -1;                  // -1 = the creature furthest ahead
    std::deque<std::string> log;
    // run record, written when the window closes (results.cpp plots it as walking.png)
    std::vector<float> rec_lives;       // per life: brain, life number, time at end (s), distance (m), died
    std::vector<float> rec_dop;         // every simulated second: time, then each brain's dopamine/s
    std::vector<float> rec_events;      // natural selection: time, replaced brain, copied brain
    double next_dop_sample = 0.0;
    // saving the population
    char save_name[64];
    std::snprintf(save_name, sizeof(save_name), "%s", ctx.name.c_str());
    std::string save_msg;
    double sim_t = 0.0;
    int step = 0;
    const float TICK = 0.01f;           // control tick: 10 ms of brain, then 10 ms of physics
    const int tick_steps = (int)std::lround(TICK * 1000.f / p.dt);

    auto end_life = [&](int b, bool fell) {
        Life& L = life[b];
        const WalkState st = world.state(b);
        L.last = st.x - L.start_x;
        if (fell) L.deaths++;
        if (fell && forfeit) L.last = std::min(L.last, 0.f);   // the dead keep no progress
        L.best = std::max(L.best, L.last);
        L.lives++; L.since_selection++;
        L.recent.push_back(L.last);
        while ((int)L.recent.size() > sel_lives) L.recent.pop_front();
        L.history.push_back(L.last);
        rec_lives.insert(rec_lives.end(), { (float)b, (float)L.lives, (float)sim_t, L.last, fell ? 1.f : 0.f });
    };
    auto new_life = [&](int b) {
        world.reset(b);
        Life& L = life[b];
        L.start_x = world.state(b).x; L.t = 0.f; L.fall_timer = -1.f;
        for (float& c : L.cmd) c = 0.f;
    };
    auto copy_brain = [&](int src, int dst) {           // dst's synapse strengths := src's
        auto seg = [&](int b, bool exc, size_t& q0, size_t& len) {
            const int g0 = exc ? b * nE : N_E + b * nI, g1 = exc ? (b + 1) * nE : N_E + (b + 1) * nI;
            q0 = (size_t)h_in_ptr[g0]; len = (size_t)(h_in_ptr[g1] - h_in_ptr[g0]);
        };
        for (int e = 0; e < 2; ++e) {
            size_t qs, ls, qd, ld;
            seg(src, e == 0, qs, ls); seg(dst, e == 0, qd, ld);
            if (ls != ld) continue;                          // brains are clones: always equal
            CUDA_CHECK(cudaMemcpy(s.in_w + qd, s.in_w + qs, ls * sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemset(s.elig + qd, 0, ld * sizeof(float)));
        }
    };

    auto tick = [&]() {
        // 1. senses -> sensory neurons (push-pull pairs of groups)
        for (int b = 0; b < K; ++b) {
            const WalkState st = world.state(b);
            float v[WALK_SENSE_GROUPS / 2 + 1];
            int k = 0;
            for (int j = 0; j < WALK_JOINTS; ++j) {
                const float lo = WalkWorld::joint_lower(j), hi = WalkWorld::joint_upper(j);
                v[k++] = (st.joint_angle[j] - 0.5f * (lo + hi)) / (0.5f * (hi - lo));
            }
            for (int j = 0; j < WALK_JOINTS; ++j) v[k++] = st.joint_speed[j] / WalkWorld::MAX_SPEED;
            v[k++] = st.angle / (death_tilt * 0.0174533f);   // full strength at the death tilt
            v[k++] = st.omega / 5.0f;
            v[k++] = st.vx / 1.5f;
            float* out = &sense[(size_t)b * nE];
            for (int g = 0; g < k; ++g) {                    // groups 0..21: + and - of 11 values
                const float x = std::max(-1.f, std::min(1.f, v[g]));
                for (int sgn = 0; sgn < 2; ++sgn) {
                    const float val = sensor_gain * std::max(0.f, sgn ? -x : x);
                    float* o = out + lay.sense_start[2 * g + sgn];
                    for (int i = 0; i < WALK_SENSE_PER; ++i) o[i] = val;
                }
            }
            for (int leg = 0; leg < 2; ++leg) {              // groups 22, 23: feet on the ground
                float* o = out + lay.sense_start[22 + leg];
                for (int i = 0; i < WALK_SENSE_PER; ++i) o[i] = st.foot_down[leg] ? sensor_gain : 0.f;
            }
            if (ctx.cpg)                                     // brainstem drive: how hard the rhythm runs
                for (int m = 0; m < WALK_MOTOR_GROUPS; ++m)
                    for (int i = 0; i < WALK_MOTOR_PER; ++i) out[lay.motor_start[m] + i] = cpg_drive;
        }
        CUDA_CHECK(cudaMemcpy(s.eye, sense.data(), sense.size() * sizeof(float), cudaMemcpyHostToDevice));
        // 2. 10 ms of brain
        rebase_steps_if_needed(s, p, step, ctx.nnz);          // long runs: keep the step counter in range
        int begin = step, end = step + tick_steps;
        void* args[] = { &s, &p, &begin, &end };
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)simulate, dim3(blocks), dim3(threads), args, 0, 0));
        step = end;
        // 3. motor neurons -> joint motors
        CUDA_CHECK(cudaMemcpy(spk.data(), s.spike_n, (size_t)N_E * sizeof(unsigned), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(s.spike_n, 0, (size_t)N * sizeof(unsigned)));
        for (int b = 0; b < K; ++b) {
            Life& L = life[b];
            const WalkState st = world.state(b);
            float cmd[WALK_JOINTS];
            for (int j = 0; j < WALK_JOINTS; ++j) {
                double r[2] = { 0, 0 };                      // flex, extend (Hz per neuron)
                for (int m = 0; m < 2; ++m) {
                    const size_t g0 = (size_t)b * nE + lay.motor_start[2 * j + m];
                    for (int i = 0; i < WALK_MOTOR_PER; ++i) r[m] += spk[g0 + i];
                    r[m] /= WALK_MOTOR_PER * TICK;
                }
                const float u = (float)((r[1] - r[0]) / motor_norm);
                L.cmd[j] += 0.3f * (u - L.cmd[j]);          // muscles smooth the commands a little
                // spinal stretch reflex: every joint is pulled back towards its resting angle
                // (hips leaned, knees straight); the brain's command adds on top of it
                const float rest = (j & 1) ? 0.f : lean;
                cmd[j] = L.cmd[j] + reflex * (rest - st.joint_angle[j]);
            }
            world.set_motors(b, cmd);
        }
        // 4. 10 ms of physics
        world.step(TICK);
        sim_t += TICK;
        if (sim_t >= next_dop_sample) {
            rec_dop.push_back((float)sim_t);
            for (int b = 0; b < K; ++b) rec_dop.push_back(life[b].dop_rate);
            next_dop_sample += 1.0;
        }
        // 5. dopamine, falls, lives
        bool any = false;
        for (int b = 0; b < K; ++b) {
            Life& L = life[b];
            const WalkState st = world.state(b);
            float a = 0.f;
            if (L.fall_timer >= 0.f) {                       // lying on the ground, about to stand up
                L.fall_timer -= TICK;
                if (L.fall_timer < 0.f) new_life(b);
            } else {
                const float d = st.vx >= 0.f ? st.vx : back_mult * st.vx;
                const float r = reward_gain * d + alive_reward;   // reward right now, per second
                if (surprise) {                                   // dopamine signals the surprise
                    a = (r - L.expect) * TICK;
                    L.expect += (TICK / expect_tau) * (r - L.expect);
                } else {
                    a = r * TICK;
                }
                L.t += TICK;
                const bool dead = st.fallen || std::fabs(st.angle) > death_tilt * 0.0174533f
                               || (kill_back && st.x - L.start_x < -kill_back_m);
                if (dead) { a -= fall_penalty; end_life(b, true); L.fall_timer = 0.5f; }
                else if (L.t >= life_len) { end_life(b, false); new_life(b); }
            }
            L.dop_rate += 0.05f * (a / TICK - L.dop_rate);
            amounts[b] = a;
            any = any || a != 0.f;
        }
        if (any) {
            CUDA_CHECK(cudaMemcpy(d_amt, amounts.data(), K * sizeof(float), cudaMemcpyHostToDevice));
            walk_reward_kernel<<<(N_E + 127) / 128, 128>>>(s.in_ptr, s.in_eend, s.in_w, s.elig, s.elig_t, N_E, nE,
                                                            step, p.k_elig, d_amt, p.w_max);
        }
        // 6. evolution: once every creature has lived sel_lives times since the last generation, rank them
        //    by mean distance over those lives; the weakest share is replaced by mutated copies of the
        //    strongest (parents dealt out in rank order); the survivors carry on unchanged (elitism).
        if (selection && K > 1) {
            bool ready = true;
            for (int b = 0; b < K; ++b) ready = ready && life[b].since_selection >= sel_lives;
            if (ready) {
                ++generation;
                auto score = [&](int b) { double s2 = 0; for (float d : life[b].recent) s2 += d; return s2 / std::max<size_t>(1, life[b].recent.size()); };
                std::vector<int> rank(K);
                for (int b = 0; b < K; ++b) rank[b] = b;
                std::stable_sort(rank.begin(), rank.end(), [&](int a, int c) { return score(a) > score(c); });
                const int n_rep = std::max(1, std::min(K - 1, (int)std::lround(replace_frac * K)));
                const int n_keep = K - n_rep;
                std::vector<double> sc(K); for (int b = 0; b < K; ++b) sc[b] = score(b);
                for (int i = 0; i < n_rep; ++i) {
                    const int dst = rank[K - 1 - i], src = rank[i % n_keep];
                    copy_brain(src, dst);
                    if (mutation > 0.f) {
                        const unsigned seed = 0x9E3779B9u * (unsigned)generation + 7919u * (unsigned)dst + 1u;
                        walk_mutate_kernel<<<(N + 255) / 256, 256>>>(s.in_ptr, s.in_eend, s.in_w, N, N_E, nE, nI,
                                                                     dst, mutation, seed, p.w_max);
                        CUDA_CHECK(cudaGetLastError());
                    }
                    rec_events.insert(rec_events.end(), { (float)sim_t, (float)dst, (float)src });
                    life[dst].recent.clear();                    // the child is judged on its own lives
                    new_life(dst);
                    char msg[200];
                    std::snprintf(msg, sizeof(msg), "gen %d: brain %d (%.2f m) <- %s copy of brain %d (%.2f m)",
                                  generation, dst + 1, sc[dst], mutation > 0.f ? "mutated" : "exact", src + 1, sc[src]);
                    log.push_front(msg);
                }
                while (log.size() > 8) log.pop_back();
                for (int b = 0; b < K; ++b) life[b].since_selection = 0;
            }
        }
    };

    for (int b = 0; b < K; ++b) new_life(b);
    double last = glfwGetTime(), sim_debt = 0.0, rt_meas = 0.0, rt_t0 = glfwGetTime(), rt_s0 = 0.0;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
        const double now = glfwGetTime();
        const double frame = std::min(0.1, now - last);
        last = now;

        // ---- advance the simulation ----
        if (fast) {
            const double until = now + 0.030;               // fill ~30 ms per frame
            while (glfwGetTime() < until) tick();
        } else {
            sim_debt = std::min(sim_debt + frame * speed, 0.2);
            const double budget_end = now + 0.030;
            while (sim_debt >= TICK && glfwGetTime() < budget_end) { tick(); sim_debt -= TICK; }
        }
        if (now - rt_t0 > 0.5) { rt_meas = (sim_t - rt_s0) / (now - rt_t0); rt_t0 = now; rt_s0 = sim_t; }

        // ---- draw ----
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        int fw, fh; glfwGetFramebufferSize(win, &fw, &fh);
        const ImVec2 disp = ImGui::GetIO().DisplaySize;
        ImDrawList* dl = ImGui::GetBackgroundDrawList();
        int cam = follow;
        if (cam < 0) { cam = 0; for (int b = 1; b < K; ++b) if (world.state(b).x > world.state(cam).x) cam = b; }
        const float scale = 160.f, ground_y = disp.y * 0.78f, cx = disp.x * 0.62f;
        const float camx = world.state(cam).x;
        auto W2S = [&](float x, float y) { return ImVec2(cx + (x - camx) * scale, ground_y - y * scale); };
        dl->AddRectFilled(ImVec2(0, ground_y), disp, IM_COL32(40, 36, 32, 255));
        dl->AddLine(ImVec2(0, ground_y), ImVec2(disp.x, ground_y), IM_COL32(150, 140, 120, 255), 2.f);
        for (int m = (int)std::floor(camx - cx / scale) - 1; m <= (int)std::ceil(camx + (disp.x - cx) / scale) + 1; ++m) {
            const ImVec2 q = W2S((float)m, 0.f);
            dl->AddLine(q, ImVec2(q.x, q.y + 10), IM_COL32(150, 140, 120, 255), 1.f);
            char lb[16]; std::snprintf(lb, sizeof(lb), "%d m", m);
            dl->AddText(ImVec2(q.x + 3, q.y + 12), IM_COL32(150, 140, 120, 255), lb);
        }
        const ImVec2 st0 = W2S(0.f, 0.f);
        dl->AddLine(ImVec2(st0.x, st0.y), ImVec2(st0.x, st0.y - 3.f * scale), IM_COL32(220, 60, 60, 160), 2.f);
        for (int pass = 0; pass < 2; ++pass)                // others first, the followed creature on top
            for (int b = 0; b < K; ++b) {
                if ((b == cam) != (pass == 1)) continue;
                const WalkState st = world.state(b);
                const ImU32 col = with_alpha(kColors[b % 8], b == cam ? 1.f : 0.35f);
                for (int q = 0; q < WALK_PARTS; ++q) {
                    const WalkPart& pt = st.part[q];
                    const float c = std::cos(pt.angle), sn = std::sin(pt.angle);
                    ImVec2 v[4];
                    const float dx[4] = { -1, 1, 1, -1 }, dy[4] = { -1, -1, 1, 1 };
                    for (int k = 0; k < 4; ++k) {
                        const float lx = dx[k] * pt.half_w, ly = dy[k] * pt.half_h;
                        v[k] = W2S(pt.x + c * lx - sn * ly, pt.y + sn * lx + c * ly);
                    }
                    dl->AddConvexPolyFilled(v, 4, col);
                }
                const ImVec2 lab = W2S(st.x, st.y + 0.25f);
                char nb[16]; std::snprintf(nb, sizeof(nb), "%d", b + 1);
                dl->AddText(lab, col, nb);
            }

        // ---- panel ----
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(470, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Walker school");
        ImGui::Text("%s brains%s", ctx.modular ? "Octopus (torso + leg modules)" : "Blob",
                    ctx.cpg ? " + spinal rhythm generators" : "");
        ImGui::Text("%d brains x %d neurons   simulated %.0f s   (%.1fx real time)", K, n, sim_t, rt_meas);
        ImGui::SliderFloat("Speed", &speed, 0.25f, 8.f, "%.2fx", ImGuiSliderFlags_Logarithmic);
        ImGui::SameLine(); ImGui::Checkbox("Fast-forward", &fast);
        ImGui::SeparatorText("Dopamine");
        ImGui::SliderFloat("Forward reward", &reward_gain, 0.f, 40.f, "%.1f");
        ImGui::SliderFloat("Backward x", &back_mult, 1.f, 5.f, "%.1f");
        ImGui::SliderFloat("Alive reward", &alive_reward, 0.f, 5.f, "%.1f /s");
        ImGui::Checkbox("Dopamine = surprise (reward minus what it expects)", &surprise);
        if (surprise) ImGui::SliderFloat("Expectation memory", &expect_tau, 1.f, 60.f, "%.0f s", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Death penalty", &fall_penalty, 0.f, 20.f, "%.1f");
        ImGui::SliderFloat("Death tilt", &death_tilt, 20.f, 90.f, "%.0f deg");
        ImGui::Checkbox("Kill if it goes backwards more than", &kill_back);
        ImGui::SameLine(); ImGui::SetNextItemWidth(110);
        ImGui::SliderFloat("##killback", &kill_back_m, 0.2f, 5.f, "%.1f m");
        ImGui::Checkbox("The dead keep no distance (score 0 m, or their loss if they went backwards)", &forfeit);
        ImGui::SliderFloat("Life length", &life_len, 5.f, 120.f, "%.0f s");
        bool da = p.da_mode != 0;
        if (ImGui::Checkbox("Learn only from dopamine", &da)) p.da_mode = da ? 1 : 0;
        if (ImGui::SliderFloat("Memory window", &elig_tau, 0.2f, 5.f, "%.1f s", ImGuiSliderFlags_Logarithmic))
            p.k_elig = p.dt / (elig_tau * 1000.f);
        ImGui::SeparatorText("Body <-> brain");
        ImGui::SliderFloat("Sensor strength", &sensor_gain, 0.05f, 1.f, "%.2f");
        ImGui::SliderFloat("Stretch reflex", &reflex, 0.f, 10.f, "%.1f");
        ImGui::SliderFloat("Posture lean", &lean, -0.4f, 0.4f, "%.2f rad");
        ImGui::SliderFloat("Muscle sensitivity", &motor_norm, 2.f, 400.f, "%.0f Hz = full", ImGuiSliderFlags_Logarithmic);
        if (ctx.cpg) {
            ImGui::SliderFloat("Rhythm drive", &cpg_drive, 0.f, 0.6f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Steady 'brainstem' input to the spinal rhythm generators.\n"
                                                          "Too low: no rhythm. Higher: stronger, faster stepping.");
        }
        ImGui::SeparatorText("Evolution");
        ImGui::Checkbox("Weakest are replaced by mutated copies of the strongest", &selection);
        ImGui::SliderInt("Lives per generation", &sel_lives, 1, 10);
        { float pct = replace_frac * 100.f;
          if (ImGui::SliderFloat("Replaced each generation", &pct, 10.f, 90.f, "%.0f%%")) replace_frac = pct / 100.f; }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Share of the population replaced (at least 1, at most all but the best)");
        { float pct = mutation * 100.f;
          if (ImGui::SliderFloat("Mutation strength", &pct, 0.f, 30.f, "%.1f%%")) mutation = pct / 100.f; }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Typical change to each synapse of a child. 0 = exact copies.");
        ImGui::Text("Generation %d", generation);
        for (const auto& m : log) ImGui::TextDisabled("%s", m.c_str());
        ImGui::SeparatorText("Brain file");
        ImGui::SetNextItemWidth(200);
        ImGui::InputText("##walkname", save_name, sizeof(save_name));
        ImGui::SameLine();
        if (ImGui::Button("Save brains")) {
            std::string name(save_name);
            if (name.empty()) name = "walkers";
            const int tag[4] = { K, n, ctx.modular ? 1 : 0, ctx.cpg ? 1 : 0 };
            save_brain("brains/" + name + ".brain", s, p, ctx.nnz, 0, nullptr, ctx.age_s + sim_t, save_msg, tag);
        }
        ImGui::TextDisabled("Saves all %d brains (closing does not save)", K);
        if (!save_msg.empty()) ImGui::TextWrapped("%s", save_msg.c_str());
        ImGui::SeparatorText("Creatures");
        ImGui::RadioButton("Follow the leader", &follow, -1);
        if (ImGui::BeginTable("cr", 7, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn(""); ImGui::TableSetupColumn("lives"); ImGui::TableSetupColumn("died"); ImGui::TableSetupColumn("last (m)");
            ImGui::TableSetupColumn("best (m)"); ImGui::TableSetupColumn("now (m)"); ImGui::TableSetupColumn("dopamine/s");
            ImGui::TableHeadersRow();
            for (int b = 0; b < K; ++b) {
                const Life& L = life[b];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::PushStyleColor(ImGuiCol_Text, kColors[b % 8]);
                char rb[24]; std::snprintf(rb, sizeof(rb), "brain %d", b + 1);
                ImGui::RadioButton(rb, &follow, b);
                ImGui::PopStyleColor();
                ImGui::TableNextColumn(); ImGui::Text("%d", L.lives);
                ImGui::TableNextColumn(); ImGui::Text("%d", L.deaths);
                ImGui::TableNextColumn(); ImGui::Text("%+.2f", L.last);
                ImGui::TableNextColumn(); if (L.lives) ImGui::Text("%+.2f", L.best); else ImGui::TextUnformatted("-");
                ImGui::TableNextColumn(); ImGui::Text("%+.2f", world.state(b).x - L.start_x);
                ImGui::TableNextColumn(); ImGui::Text("%+.2f", L.dop_rate);
            }
            ImGui::EndTable();
        }
        if (ImGui::Button("Stand everyone up")) for (int b = 0; b < K; ++b) new_life(b);
        if (ImPlot::BeginPlot("Distance per life (m)", ImVec2(-1, 200))) {
            ImPlot::SetupAxes("life", nullptr, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            for (int b = 0; b < K; ++b) {
                if (life[b].history.empty()) continue;
                ImPlotSpec sp; sp.LineColor = ImGui::ColorConvertU32ToFloat4(kColors[b % 8]); sp.LineWeight = 1.5f;
                char nm[24]; std::snprintf(nm, sizeof(nm), "brain %d", b + 1);
                ImPlot::PlotLine(nm, life[b].history.data(), (int)life[b].history.size(), 1.0, 1.0, sp);
            }
            ImPlot::EndPlot();
        }
        ImGui::End();

        ImGui::Render();
        glViewport(0, 0, fw, fh);
        glClearColor(0.08f, 0.09f, 0.12f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_amt);
    // ---- write the run record for the results figure ----
    {
        auto wbin = [](const char* name, const std::vector<float>& v) {
            if (FILE* f = std::fopen(name, "wb")) { if (!v.empty()) std::fwrite(v.data(), sizeof(float), v.size(), f); std::fclose(f); }
        };
        wbin("walk_lives.bin", rec_lives);
        wbin("walk_dopamine.bin", rec_dop);
        wbin("walk_events.bin", rec_events);
        if (FILE* f = std::fopen("meta.txt", "w")) {
            std::fprintf(f, "version 3\nwalk 1\nbrains %d\nn %d\nmodular %d\nsim_s %.3f\nage_s %.3f\ndt %g\nsyn %g\n"
                            "reward %g\nback_mult %g\nalive %g\nsurprise %d\ndeath_penalty %g\ndeath_tilt %g\n"
                            "life_len %g\nreflex %g\nlean %g\nmuscle_hz %g\nselection %d\ngenerations %d\nmutation %g\nreplace_frac %g\n"
                            "cpg %d\ncpg_drive %g\n",
                         K, n, ctx.modular ? 1 : 0, sim_t, ctx.age_s, p.dt, p.k_local, reward_gain, back_mult,
                         alive_reward, surprise ? 1 : 0, fall_penalty, death_tilt, life_len, reflex, lean, motor_norm,
                         selection ? 1 : 0, generation, mutation, replace_frac, ctx.cpg ? 1 : 0, cpg_drive);
            std::fclose(f);
        }
        std::printf("Walkers: %.0f s simulated, %zu lives. Wrote meta.txt and walk_*.bin\n", sim_t, rec_lives.size() / 5);
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

#else
int run_walk(DevState&, Params&, int, int, const WalkCtx&) {
    std::fprintf(stderr, "The walking creatures need the visualizer with its panel (BRAIN_VIZ and BRAIN_VIZ_UI).\n");
    return 1;
}
#endif
