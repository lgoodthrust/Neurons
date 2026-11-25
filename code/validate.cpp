// validate.cpp - independent CPU reference for the GPU simulation (ported from brain.py).
//
// This file deliberately shares NO code with the CUDA kernel: the model is written out
// again here, step by step, from the equations, and reads the network from the output
// files. The GPU run (float) is compared against this reference in double precision; the
// same reference in float gives the size of pure rounding noise. A GPU error far above
// that noise means a bug.
#include "validate.h"
#include "files.h"
#include "plotwin.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>
#include <imgui.h>
#include <implot.h>

namespace {

const double VALIDATE_REPLAY_MS = 2000;   // only the start of a run is replayed

struct Net {
    int N = 0, N_E = 0, N_A = 0, B = 4, steps = 0;
    int trace_ids[3] = { 0, 0, 0 };
    // synapses in incoming order (grouped by postsynaptic neuron, E inputs first)
    std::vector<int> ptr, eend, pre, post, delay, branch;
    std::vector<double> w0;
    std::vector<char> plastic;
    std::vector<int> out_start, out_idx;   // synapses grouped by presynaptic neuron
    std::vector<float> Vs0;
    void build_out() {
        out_start.assign(N + 1, 0);
        for (int p : pre) out_start[p + 1]++;
        for (int n = 0; n < N; ++n) out_start[n + 1] += out_start[n];
        out_idx.assign(pre.size(), 0);
        std::vector<int> fill(out_start.begin(), out_start.end() - 1);
        for (size_t q = 0; q < pre.size(); ++q) out_idx[fill[pre[q]]++] = (int)q;
    }
};

struct Run {                                // traces [step][trace] and [step][trace][branch]
    int steps = 0, NT = 3, B = 4;
    std::vector<double> tr_s, tr_b;
    std::vector<std::pair<int, int>> spikes;   // (step, neuron), sorted
    std::vector<double> W;                     // final weight per synapse (empty = not available)
};

// ---------------------------------------------------------------------------
//  Reference simulation (mirrors the model step by step), T = double or float.
//  ALT = true writes the exponential decays as x *= (1 - dt/tau) instead of
//  x -= x/tau*dt: the same maths with different rounding (used by the self-test).
// ---------------------------------------------------------------------------
template <class T, bool ALT = false>
Run simulate(const Net& net, const KV& p, bool progress) {
    const int N = net.N, N_E = net.N_E, N_A = net.N_A, B = net.B, steps = net.steps;
    const int L = (int)kv(p, "L");
    auto f = [&](const char* k) { return (T)kv(p, k); };
    const T dt = f("dt");
    const T EL = f("EL"), VT = f("VT"), Vpk = f("V_peak"), Eexc = f("E_exc"), Einh = f("E_inh");
    const T gexc = f("g_exc"), ginh = f("g_inh"), ratio = f("nmda_ratio"), Mg = f("Mg");
    const T Cb = f("Cb"), gLb = f("gLb"), g_cb = f("g_cb"), bAP = f("bAP");
    const T tau_ampa = f("tau_ampa"), tau_nr = f("tau_nr"), tau_nd = f("tau_nd"), tau_gaba = f("tau_gaba");
    const T tau_plus = f("tau_plus"), tau_minus = f("tau_minus"), A_plus = f("A_plus"), A_minus = f("A_minus");
    const T w_max = f("w_max"), stim_lam = f("stim_lam"), lamI = f("lamI");
    const int stim_every = (int)kv(p, "stim_every"), stim_len = (int)kv(p, "stim_len");

    std::vector<T> Cs(N), gLs(N), dT(N), Vr(N), a(N), bw(N), tau_w(N), t_ref(N), frac_s(N), lamE(N);
    for (int n = 0; n < N; ++n) {
        const bool e = n < N_E;
        Cs[n] = e ? f("Cs_E") : f("Cs_I");   gLs[n] = e ? f("gLs_E") : f("gLs_I");
        dT[n] = e ? f("dT_E") : f("dT_I");   Vr[n] = e ? f("Vr_E") : f("Vr_I");
        a[n] = e ? f("a_E") : T(0);          bw[n] = e ? f("b_E") : T(0);
        tau_w[n] = e ? f("tauw_E") : T(1);   t_ref[n] = e ? f("tref_E") : f("tref_I");
        frac_s[n] = e ? f("inh_soma_frac") : T(1);
        lamE[n] = e ? f("lamE_branch_E") : f("lamE_branch_I");
    }
    std::vector<T> W(net.w0.size());
    for (size_t q = 0; q < W.size(); ++q) W[q] = (T)net.w0[q];

    std::vector<T> Vs(N), w(N, 0), ref(N, 0), g_gaba(N, 0), x_pre(N, 0), y_post(N, 0);
    std::vector<T> Vb((size_t)N * B), g_ampa((size_t)N * B, 0), n_r((size_t)N * B, 0), n_d((size_t)N * B, 0);
    for (int n = 0; n < N; ++n) { Vs[n] = (T)net.Vs0[n]; for (int k = 0; k < B; ++k) Vb[(size_t)n * B + k] = Vs[n]; }
    std::vector<T> buf_exc((size_t)L * N * B, 0), buf_inh((size_t)L * N, 0);
    std::vector<char> spk(N, 0);
    std::vector<int> J;
    auto mg = [&](T V) { return T(1) / (T(1) + (Mg / T(3.57)) * std::exp(T(-0.062) * V)); };

    Run out; out.steps = steps; out.B = B;
    out.tr_s.assign((size_t)steps * 3, 0.0); out.tr_b.assign((size_t)steps * 3 * B, 0.0);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<T> dVb(B);

    for (int step = 0; step < steps; ++step) {
        const int slot = step % L;
        const bool stim_on = step % stim_every < stim_len;
        // ---- phase A: every neuron integrates one step ----
        for (int n = 0; n < N; ++n) {
            const bool e = n < N_E;
            T* gA = &g_ampa[(size_t)n * B]; T* nr = &n_r[(size_t)n * B]; T* nd = &n_d[(size_t)n * B];
            T* vb = &Vb[(size_t)n * B];
            T* be = &buf_exc[((size_t)slot * N + n) * B];
            for (int k = 0; k < B; ++k) {
                T ext = lamE[n];
                if (k == 0 && stim_on && n < N_A) ext += stim_lam;
                const T inp = be[k] + ext * gexc;
                be[k] = 0;
                gA[k] += inp; nr[k] += inp * ratio; nd[k] += inp * ratio;
            }
            g_gaba[n] += buf_inh[(size_t)slot * N + n] + lamI * ginh;
            buf_inh[(size_t)slot * N + n] = 0;

            T I_soma = frac_s[n] * g_gaba[n] * (Einh - Vs[n]);
            T sum_cpl = 0, sum_exc = 0;
            for (int k = 0; k < B; ++k) {
                const T Vsite = e ? vb[k] : Vs[n];
                const T gn = (nd[k] - nr[k]) * mg(Vsite);
                const T Iexc = (gA[k] + gn) * (Eexc - Vsite);
                const T Icpl = g_cb * (vb[k] - Vs[n]);
                const T Iinhb = ((T(1) - frac_s[n]) * g_gaba[n] / (T)B) * (Einh - vb[k]);
                sum_cpl += Icpl; sum_exc += Iexc;
                dVb[k] = (-gLb * (vb[k] - EL) - Icpl + Iexc + Iinhb) / Cb * dt;
            }
            I_soma += e ? sum_cpl : sum_exc;
            const T expt = gLs[n] * dT[n] * std::exp(std::min((Vs[n] - VT) / dT[n], T(20)));
            const T dVs = (-gLs[n] * (Vs[n] - EL) + expt - w[n] + I_soma) / Cs[n] * dt;
            const T dw = (a[n] * (Vs[n] - EL) - w[n]) / tau_w[n] * dt;
            if (ref[n] <= 0) Vs[n] += dVs;
            for (int k = 0; k < B; ++k) vb[k] = e ? vb[k] + dVb[k] : Vs[n];
            w[n] += dw;
            ref[n] -= dt;
            if (!ALT) {
                for (int k = 0; k < B; ++k) {
                    gA[k] -= gA[k] / tau_ampa * dt; nr[k] -= nr[k] / tau_nr * dt; nd[k] -= nd[k] / tau_nd * dt;
                }
                g_gaba[n] -= g_gaba[n] / tau_gaba * dt;
                x_pre[n] -= x_pre[n] / tau_plus * dt;
                y_post[n] -= y_post[n] / tau_minus * dt;
            } else {
                for (int k = 0; k < B; ++k) {
                    gA[k] *= T(1) - dt / tau_ampa; nr[k] *= T(1) - dt / tau_nr; nd[k] *= T(1) - dt / tau_nd;
                }
                g_gaba[n] *= T(1) - dt / tau_gaba;
                x_pre[n] *= T(1) - dt / tau_plus;
                y_post[n] *= T(1) - dt / tau_minus;
            }
            spk[n] = Vs[n] >= Vpk;
        }
        // ---- phase B: spikes ----
        J.clear();
        for (int n = 0; n < N; ++n) if (spk[n]) J.push_back(n);
        for (int j : J) {                          // deliver with the old weight, then depress (pre after post)
            for (int o = net.out_start[j]; o < net.out_start[j + 1]; ++o) {
                const int q = net.out_idx[o], po = net.post[q];
                const int arr = (step + net.delay[q]) % L;
                const T wq = W[q];
                if (j < N_E) buf_exc[((size_t)arr * N + po) * B + net.branch[q]] += wq;
                else         buf_inh[(size_t)arr * N + po] += wq;
                if (net.plastic[q]) W[q] = std::max(wq - A_minus * y_post[po], T(0));
            }
        }
        for (int po : J) {                         // excitatory cells that spiked: potentiate their inputs
            if (po >= N_E) continue;
            for (int q = net.ptr[po]; q < net.eend[po]; ++q)
                if (net.plastic[q]) W[q] = std::min(W[q] + A_plus * x_pre[net.pre[q]], w_max);
        }
        for (int j : J) {
            x_pre[j] += 1; y_post[j] += 1;
            Vs[j] = Vr[j];
            if (j < N_E) for (int k = 0; k < B; ++k) Vb[(size_t)j * B + k] += bAP;
            w[j] += bw[j]; ref[j] = t_ref[j];
            out.spikes.push_back({ step, j });
        }
        for (int k = 0; k < 3; ++k) {
            const int tid = net.trace_ids[k];
            out.tr_s[(size_t)step * 3 + k] = spk[tid] ? (double)Vpk : (double)Vs[tid];
            for (int b = 0; b < B; ++b) out.tr_b[((size_t)step * 3 + k) * B + b] = (double)Vb[(size_t)tid * B + b];
        }
        if (progress && step % 2000 == 0 && step) {
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("\r  reference: %d/%d steps  (%.0f s, ~%.0f s left)   ", step, steps, el, el / step * (steps - step));
            std::fflush(stdout);
        }
    }
    if (progress) {
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("\r  reference: %d/%d steps  (%.1f s)                    \n", steps, steps, el);
    }
    out.W.assign(W.begin(), W.end());
    return out;
}

// ---------------------------------------------------------------------------
//  Comparison
// ---------------------------------------------------------------------------
// earliest step at which two sorted spike lists differ (-1 = identical), and how many matched
std::pair<int, size_t> first_spike_mismatch(const Run& A, const Run& Bref) {
    const auto& sa = A.spikes; const auto& sb = Bref.spikes;
    const size_t k = std::min(sa.size(), sb.size());
    for (size_t i = 0; i < k; ++i)
        if (sa[i] != sb[i]) return { std::min(sa[i].first, sb[i].first), i };
    if (sa.size() != sb.size()) return { (sa.size() > k ? sa : sb)[k].first, k };
    return { -1, k };
}

double percentile(std::vector<double> v, double pct) {      // numpy's default (linear)
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    const double pos = pct / 100.0 * (v.size() - 1);
    const size_t lo = (size_t)pos, hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (pos - lo) * (v[hi] - v[lo]);
}

// 99th percentile |dV| over soma + branch traces of all recorded neurons, steps [0, end)
double trace_error(const Run& A, const Run& Bref, int end) {
    if (end <= 1) return NAN;
    std::vector<double> d;
    d.reserve((size_t)end * 3 * (1 + A.B));
    for (size_t i = 0; i < (size_t)end * 3; ++i) d.push_back(std::fabs(A.tr_s[i] - Bref.tr_s[i]));
    for (size_t i = 0; i < (size_t)end * 3 * A.B; ++i) d.push_back(std::fabs(A.tr_b[i] - Bref.tr_b[i]));
    return percentile(d, 99);
}

int clean_end(const Run& A, const Run& Bref, int steps, double dt) {
    const int m = first_spike_mismatch(A, Bref).first;
    return m < 0 ? steps : std::max(m - (int)(2.0 / dt), 0);
}

struct PlotData { std::vector<double> t, ref_s[3], gpu_s[3], ref_b[2], gpu_b[2], err[3]; double mis_ms = -1; };

bool compare(const Net& net, const KV& p, const Run& A, const Run& Bref, const char* la, const char* lb,
             const Run* baseline, PlotData* pd) {
    const double dt = kv(p, "dt");
    const int steps = net.steps, N = net.N, N_E = net.N_E, N_A = net.N_A;
    char names[3][64];
    std::snprintf(names[0], 64, "neuron %d (group A, E)", net.trace_ids[0]);
    std::snprintf(names[1], 64, "neuron %d (other E)", net.trace_ids[1]);
    std::snprintf(names[2], 64, "neuron %d (inhibitory)", net.trace_ids[2]);

    const auto mm = first_spike_mismatch(A, Bref);
    const int mis = mm.first;
    const int guard = (int)(2.0 / dt);              // voltages compared until 2 ms before the first spike mismatch
    const int end = mis >= 0 ? std::max(mis - guard, 0) : steps;
    std::printf("\n================ VALIDATION REPORT ================\n");
    std::printf("Spikes: %s %zu, %s %zu\n", la, A.spikes.size(), lb, Bref.spikes.size());
    if (mis < 0) std::printf("Spike times: IDENTICAL for the whole run\n");
    else std::printf("Spike times: identical for the first %.1f ms (%zu spikes), first mismatch at step %d\n",
                     mis * dt, mm.second, mis);

    std::printf("\nVoltage agreement until 2 ms before the first spike mismatch (0 - %.1f ms):\n", end * dt);
    for (int k = 0; k < 3 && end > 1; ++k) {
        double ds = 0, db = 0;
        for (int s = 0; s < end; ++s) {
            ds = std::max(ds, std::fabs(A.tr_s[(size_t)s * 3 + k] - Bref.tr_s[(size_t)s * 3 + k]));
            for (int b = 0; b < A.B; ++b) {
                const size_t i = ((size_t)s * 3 + k) * A.B + b;
                db = std::max(db, std::fabs(A.tr_b[i] - Bref.tr_b[i]));
            }
        }
        // Inhibitory cells are single-compartment: their branch slots only copy the soma, including
        // the one-step spike overshoot before the reset, so a branch comparison means nothing there.
        if (k == 2) std::printf("  %-26s soma max |dV| %.2e mV   branches n/a (single compartment)\n", names[k], ds);
        else        std::printf("  %-26s soma max |dV| %.2e mV   branches max |dV| %.2e mV\n", names[k], ds, db);
    }
    if (end < steps) std::printf("\nAfter divergence (%.0f - %.0f ms), statistics should still agree:\n", end * dt, steps * dt);
    const int lo = (int)std::min((double)end, steps * 0.5);
    auto rates = [&](const Run& R, double out[3]) {
        double c[3] = { 0, 0, 0 };
        for (auto& sp : R.spikes) if (sp.first >= lo) c[sp.second < N_A ? 0 : sp.second < N_E ? 1 : 2] += 1;
        const double dur = std::max((steps - lo) * dt / 1000.0, 1e-9);
        out[0] = c[0] / N_A / dur; out[1] = c[1] / (N_E - N_A) / dur; out[2] = c[2] / (N - N_E) / dur;
    };
    double ra[3], rb[3]; rates(A, ra); rates(Bref, rb);
    std::printf("  firing rates from %.0f ms (group A / other E / I):\n", lo * dt);
    std::printf("    %-18s %6.2f / %6.2f / %6.2f Hz\n", la, ra[0], ra[1], ra[2]);
    std::printf("    %-18s %6.2f / %6.2f / %6.2f Hz\n", lb, rb[0], rb[1], rb[2]);

    if (A.W.empty()) {
        std::printf("\nSTDP final weights: skipped (only part of the run was replayed)\n");
    } else {
        double sa = 0, sb = 0, dmax = 0; size_t n = 0;
        for (size_t q = 0; q < A.W.size(); ++q) if (net.plastic[q]) {
            sa += A.W[q]; sb += Bref.W[q]; dmax = std::max(dmax, std::fabs(A.W[q] - Bref.W[q])); ++n;
        }
        std::printf("\nSTDP final E->E weights: mean %.4f vs %.4f nS, max |dW| %.2e nS\n", sa / n, sb / n, dmax);
        std::printf(mis < 0 ? "  (no divergence, so this is a direct check of the STDP updates)\n"
                            : "  (runs diverged, so compare the means, not individual synapses)\n");
    }

    const bool early_bad = mis >= 0 && mis * dt < 1.0;
    bool v_bad;
    if (baseline) {                                 // baseline = the same network in float: pure rounding noise
        const int win = std::min(end, clean_end(*baseline, Bref, steps, dt));
        const double e_gpu = trace_error(A, Bref, win), e_base = trace_error(*baseline, Bref, win);
        const double ratio = e_gpu / std::max(e_base, 1e-12);
        std::printf("\nError vs rounding-only baseline (0 - %.1f ms, 99th pct |dV|):\n", win * dt);
        std::printf("  %s: %.2e mV   CPU float: %.2e mV   ratio %.1fx\n", la, e_gpu, e_base, ratio);
        v_bad = !(ratio <= 30);
    } else {
        v_bad = !(trace_error(A, Bref, end) <= 1e-2);
    }
    if (end <= 1 || !std::isfinite(trace_error(A, Bref, end))) {
        v_bad = true;
        std::printf("  Insufficient pre-divergence samples for reliable voltage comparison\n");
    }
    const bool passed = !(early_bad || v_bad);
    std::printf("\nVERDICT: %s\n===================================================\n",
                passed ? "PASS - differences consistent with float32 rounding" : "LIKELY BUG / INCONCLUSIVE - investigate");

    if (pd) {
        const int span = std::min(steps, std::max((int)(300 / dt), std::min(end + (int)(100 / dt), steps)));
        pd->mis_ms = mis >= 0 ? mis * dt : -1;
        for (int s = 0; s < span; ++s) {
            pd->t.push_back(s * dt);
            for (int k = 0; k < 3; ++k) {
                pd->ref_s[k].push_back(Bref.tr_s[(size_t)s * 3 + k]);
                pd->gpu_s[k].push_back(A.tr_s[(size_t)s * 3 + k]);
                pd->err[k].push_back(std::fabs(A.tr_s[(size_t)s * 3 + k] - Bref.tr_s[(size_t)s * 3 + k]) + 1e-9);
            }
            for (int k = 0; k < 2; ++k) {
                pd->ref_b[k].push_back(Bref.tr_b[((size_t)s * 3 + k) * A.B]);
                pd->gpu_b[k].push_back(A.tr_b[((size_t)s * 3 + k) * A.B]);
            }
        }
    }
    return passed;
}

ImPlotSpec ls(ImVec4 c, float w) { ImPlotSpec s; s.LineColor = c; s.LineWeight = w; return s; }

void plot_validation(const PlotData& pd, const char* names[3], const std::string& head, bool passed) {
    const std::string title = head + "   |   verdict: " + (passed ? "PASS" : "FAIL / INCONCLUSIVE");
    auto draw = [&]() {
        ImGui::TextUnformatted(title.c_str());
        const float H = plot_avail_h() / 4 - 4;
        const int n = (int)pd.t.size();
        for (int k = 0; k < 3; ++k) {
            if (ImPlot::BeginPlot(names[k], ImVec2(-1, H))) {
                ImPlot::SetupAxes(nullptr, "mV", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupLegend(ImPlotLocation_NorthEast);
                ImPlot::PlotLine("soma, reference (double)", pd.t.data(), pd.ref_s[k].data(), n, ls(ImVec4(0.95f, 0.95f, 0.95f, 1), 2.4f));
                ImPlot::PlotLine("soma, GPU (float)", pd.t.data(), pd.gpu_s[k].data(), n, ls(ImVec4(0.9f, 0.15f, 0.15f, 1), 1.0f));
                if (k < 2) {
                    ImPlot::PlotLine("branch 0, reference", pd.t.data(), pd.ref_b[k].data(), n, ls(ImVec4(1.0f, 0.5f, 0.05f, 1), 2.0f));
                    ImPlot::PlotLine("branch 0, GPU", pd.t.data(), pd.gpu_b[k].data(), n, ls(ImVec4(0.2f, 0.5f, 0.95f, 1), 0.9f));
                }
                if (pd.mis_ms >= 0) {
                    ImPlotSpec s = ls(ImVec4(0.6f, 0.6f, 0.6f, 1), 1.0f); s.Flags = ImPlotItemFlags_NoLegend;
                    ImPlot::PlotInfLines("##mis", &pd.mis_ms, 1, s);
                }
                ImPlot::EndPlot();
            }
        }
        if (ImPlot::BeginPlot("|soma dV| (mV), log scale", ImVec2(-1, H))) {
            ImPlot::SetupAxes("time (ms)", "|dV| (mV)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
            ImPlot::SetupLegend(ImPlotLocation_SouthEast);
            const ImVec4 c[3] = { ImVec4(0.12f, 0.47f, 0.71f, 1), ImVec4(1.0f, 0.5f, 0.05f, 1), ImVec4(0.17f, 0.63f, 0.17f, 1) };
            for (int k = 0; k < 3; ++k) ImPlot::PlotLine(names[k], pd.t.data(), pd.err[k].data(), n, ls(c[k], 1.0f));
            if (pd.mis_ms >= 0) ImPlot::PlotInfLines("first spike mismatch", &pd.mis_ms, 1, ls(ImVec4(0.6f, 0.6f, 0.6f, 1), 1.0f));
            ImPlot::EndPlot();
        }
    };
    plot_window("Brain - validation", 1500, 1000, draw, "validation.png");
}

std::string Pth(const std::string& folder, const char* name) {
    return folder.empty() ? std::string(name) : folder + "/" + name;
}

}  // namespace

bool validate_output(const std::string& folder, bool show_window) {
    const KV meta = read_kv(Pth(folder, "meta.txt")), prm = read_kv(Pth(folder, "params.txt"));
    if (meta.empty() || prm.empty()) { std::printf("No output to validate - run in validate mode first\n"); return false; }
    Net net;
    net.N = (int)kv(meta, "N"); net.N_E = (int)kv(meta, "N_E"); net.N_A = (int)kv(meta, "N_A");
    net.B = (int)kv(meta, "B"); net.steps = (int)kv(meta, "steps");
    const int NT = (int)kv(meta, "n_trace"), n_rec = (int)kv(meta, "n_rec");
    for (int k = 0; k < 3; ++k) net.trace_ids[k] = (int)kv(meta, "trace_id" + std::to_string(k));
    if (net.N > 5000) { std::printf("The reference is meant for small networks (got N = %d); validate with N = 1000.\n", net.N); return false; }
    if ((int)kv(meta, "mode", 0) != 1 || (int)kv(prm, "rec_every") != 1) {
        std::printf("The latest output is not a validate-mode run.\n"); return false;
    }
    net.ptr = read_bin<int>(Pth(folder, "csr_ptr.bin")); net.eend = read_bin<int>(Pth(folder, "csr_eend.bin"));
    net.pre = read_bin<int>(Pth(folder, "csr_pre.bin"));
    auto meta16 = read_bin<uint16_t>(Pth(folder, "csr_meta.bin"));
    auto w0 = read_bin<float>(Pth(folder, "csr_w_init.bin")), w1 = read_bin<float>(Pth(folder, "csr_w_final.bin"));
    if ((int)net.ptr.size() != net.N + 1 || net.pre.empty() || meta16.size() != net.pre.size() || w0.size() != net.pre.size()) {
        std::printf("Synapse files missing: run in validate mode first.\n"); return false;
    }
    const size_t nnz = net.pre.size();
    net.post.resize(nnz); net.delay.resize(nnz); net.branch.resize(nnz); net.w0.resize(nnz); net.plastic.resize(nnz);
    for (int po = 0; po < net.N; ++po)
        for (int q = net.ptr[po]; q < net.ptr[po + 1]; ++q) {
            net.post[q] = po; net.delay[q] = meta16[q] & 63; net.branch[q] = meta16[q] >> 6; net.w0[q] = w0[q];
            net.plastic[q] = q < net.eend[po] && po < net.N_E;          // E inputs onto E cells
        }
    net.build_out();
    net.Vs0 = read_bin<float>(Pth(folder, "Vs0.bin"));

    // the GPU's run
    Run gpu; gpu.B = net.B;
    auto trs = read_bin<float>(Pth(folder, "trace_soma.bin")), trb = read_bin<float>(Pth(folder, "trace_branch.bin"));
    if ((int)trs.size() < n_rec * NT || NT != 3) { std::printf("Trace files missing.\n"); return false; }
    auto sp = read_bin<int>(Pth(folder, "spikes.bin"));
    const double dt = kv(prm, "dt");
    const int replay = std::min(net.steps, (int)std::lround(VALIDATE_REPLAY_MS / dt));
    if (replay < net.steps) std::printf("\nReplaying the first %.0f ms of the run on the CPU\n", replay * dt);
    gpu.steps = replay;
    gpu.tr_s.assign(trs.begin(), trs.begin() + (size_t)replay * 3);
    gpu.tr_b.assign(trb.begin(), trb.begin() + (size_t)replay * 3 * net.B);
    for (size_t k = 0; k + 1 < sp.size(); k += 2) if (sp[k] < replay) gpu.spikes.push_back({ sp[k], sp[k + 1] });
    std::sort(gpu.spikes.begin(), gpu.spikes.end());
    if (replay == net.steps) gpu.W.assign(w1.begin(), w1.end());
    net.steps = replay;

    std::printf("\nRe-running %d neurons for %.0f ms in double precision...\n", net.N, net.steps * dt);
    Run ref = simulate<double>(net, prm, true);
    std::printf("Running the same network in float for a rounding-noise baseline...\n");
    Run base = simulate<float>(net, prm, true);
    if (gpu.W.empty()) ref.W.clear();

    const double T_s = kv(meta, "T_ms") / 1000.0, wall = kv(meta, "wall_s");
    char head[256];
    std::snprintf(head, sizeof(head), "Validation: %d neurons, %.1f s simulated in %.2f s on the GPU (%.1fx real time)",
                  net.N, T_s, wall, T_s / wall);
    PlotData pd;
    const bool passed = compare(net, prm, gpu, ref, "GPU float32", "reference double", &base, &pd);
    if (show_window) {
        char n0[64], n1[64], n2[64];
        std::snprintf(n0, 64, "neuron %d (group A, E)", net.trace_ids[0]);
        std::snprintf(n1, 64, "neuron %d (other E)", net.trace_ids[1]);
        std::snprintf(n2, 64, "neuron %d (inhibitory)", net.trace_ids[2]);
        const char* names[3] = { n0, n1, n2 };
        plot_validation(pd, names, head, passed);
    }
    return passed;
}

bool validator_selftest() {
    // synthetic network: 1,000 neurons, dense random connectivity, same model parameters
    std::mt19937 rng(2);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double s2 = std::log(2.0);
    std::lognormal_distribution<double> LN(-s2 / 2, std::sqrt(s2));
    std::uniform_int_distribution<int> Dd(10, 30), Bd(0, 3);
    const double g_exc = 0.35, g_inh = 0.8;
    Net net;
    net.N = 1000; net.N_E = 800; net.N_A = 200; net.B = 4; net.steps = 3000;
    net.trace_ids[0] = 0; net.trace_ids[1] = net.N_A; net.trace_ids[2] = net.N_E;
    net.ptr.assign(1, 0);
    for (int po = 0; po < net.N; ++po) {
        for (int pass = 0; pass < 2; ++pass) {               // E inputs first, then I
            for (int pr = pass ? net.N_E : 0; pr < (pass ? net.N : net.N_E); ++pr) {
                if (pr == po || U(rng) >= (pr < net.N_E ? 0.1 : 0.4)) continue;
                const bool pl = pr < net.N_E && po < net.N_E;
                double w = LN(rng) * (pr < net.N_E ? g_exc : g_inh);
                if (pl) w = std::min(w, 8 * g_exc);
                net.pre.push_back(pr); net.post.push_back(po); net.delay.push_back(Dd(rng));
                net.branch.push_back(Bd(rng)); net.w0.push_back((float)w); net.plastic.push_back(pl);
            }
            if (pass == 0) net.eend.push_back((int)net.pre.size());
        }
        net.ptr.push_back((int)net.pre.size());
    }
    net.build_out();
    for (int n = 0; n < net.N; ++n) net.Vs0.push_back((float)(-70 + 10 * U(rng)));
    const double dt = 0.1, rate = 3.9, B = 4;
    KV p = { {"dt", dt}, {"Cs_E", 100}, {"Cs_I", 200}, {"gLs_E", 5}, {"gLs_I", 20}, {"EL", -70}, {"VT", -50},
             {"dT_E", 2}, {"dT_I", 0.5}, {"Vr_E", -58}, {"Vr_I", -60}, {"a_E", 2}, {"b_E", 60}, {"tauw_E", 200},
             {"tref_E", 2}, {"tref_I", 1}, {"V_peak", 0}, {"Cb", 100}, {"gLb", 10}, {"g_cb", 30}, {"bAP", 8},
             {"inh_soma_frac", 0.3}, {"E_exc", 0}, {"E_inh", -80}, {"tau_ampa", 5}, {"tau_gaba", 10},
             {"tau_nr", 2}, {"tau_nd", 100}, {"nmda_ratio", 0.1}, {"Mg", 1}, {"g_exc", g_exc}, {"g_inh", g_inh},
             {"lamE_branch_E", (5600 - 80) * rate * dt / 1000 / B},
             {"lamE_branch_I", (5600 - 80) * rate * dt / 1000 / B * 0.55},
             {"lamI", (1300 - 80) * 2.4 * dt / 1000}, {"stim_lam", 40 * dt}, {"tau_plus", 17}, {"tau_minus", 34},
             {"A_plus", 0.01 * g_exc}, {"A_minus", 0.0053 * g_exc}, {"w_max", 8 * g_exc}, {"L", 31},
             {"stim_every", 2000}, {"stim_len", 100}, {"warmup", 0}, {"rec_every", 1}, {"w_every", 2500} };
    // Same test as the GPU gets: a float candidate vs the double reference, judged against the
    // rounding noise of a plain float run. The correct candidate is a float run with different
    // (but mathematically equal) rounding; the broken one has the wrong NMDA decay.
    std::printf("Self-test: synthetic network, 300 ms, float candidates vs double reference\n");
    const Run b = simulate<double>(net, p, false);
    const Run a = simulate<float>(net, p, false);
    const Run alt = simulate<float, true>(net, p, false);
    std::printf("\n--- correct model, different float rounding (should PASS) ---\n");
    const bool good_pass = compare(net, p, alt, b, "float, other rounding", "double", &a, nullptr);
    KV pb = p; pb["tau_nd"] = 90.0;
    const Run bug = simulate<float>(net, pb, false);
    std::printf("\n--- deliberately broken model (NMDA decay 90 ms instead of 100) vs double (should FAIL) ---\n");
    const bool bad_pass = compare(net, p, bug, b, "broken float", "double", &a, nullptr);
    if (!good_pass || bad_pass) { std::printf("SELFTEST FAILED: expected good=PASS and bad=FAIL\n"); return false; }
    std::printf("SELFTEST PASSED: correct model accepted; deliberate NMDA bug rejected\n");
    return true;
}
