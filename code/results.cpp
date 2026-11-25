// results.cpp - statistics and figures for Brain's output files (ported from brain.py)
#include "results.h"
#include "files.h"
#include "plotwin.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>
#include <imgui.h>
#include <implot.h>

static std::string P(const std::string& folder, const char* name) {
    return folder.empty() ? std::string(name) : folder + "/" + name;
}

static ImPlotSpec line_spec(ImVec4 col, float weight = 1.0f) {
    ImPlotSpec s; s.LineColor = col; s.LineWeight = weight; return s;
}
static ImPlotSpec dot_spec(ImVec4 col, float size) {
    ImPlotSpec s; s.Marker = ImPlotMarker_Circle; s.MarkerSize = size;
    s.MarkerFillColor = col; s.MarkerLineColor = col; s.LineWeight = 0.0f; return s;
}
static ImPlotSpec fill_spec(ImVec4 col, float alpha) {
    ImPlotSpec s; s.FillColor = col; s.FillAlpha = alpha; s.LineWeight = 0.0f; return s;
}
// shaded vertical band [x0, x1] from y0 to y1
static void band(const char* id, double x0, double x1, double y0, double y1, ImVec4 col, float alpha) {
    const double xs[2] = { x0, x1 }, ys[2] = { y1, y1 };
    ImPlotSpec s = fill_spec(col, alpha);
    s.Flags = ImPlotItemFlags_NoLegend;
    ImPlot::PlotShaded(id, xs, ys, 2, y0, s);
}

static const ImVec4 C_GREEN(0.17f, 0.63f, 0.17f, 1), C_BLUE(0.12f, 0.47f, 0.71f, 1), C_RED(0.84f, 0.15f, 0.16f, 1);
static const ImVec4 C_GRAY(0.6f, 0.6f, 0.6f, 1), C_WHITE(0.92f, 0.92f, 0.92f, 1);

// =============================================================================
//  Normal run
// =============================================================================
void plot_run(const std::string& folder, bool show_window) {
    if (!file_exists(P(folder, "meta.txt"))) { std::printf("meta.txt not found - run a simulation first\n"); return; }
    const KV meta = read_kv(P(folder, "meta.txt"));
    if (kv(meta, "version", 1) < 3) { std::printf("This plotter needs Brain.exe v3+ output.\n"); return; }
    const int NT = (int)kv(meta, "n_trace"), N = (int)kv(meta, "N"), N_E = (int)kv(meta, "N_E");
    const int N_A = (int)kv(meta, "N_A"), B = (int)kv(meta, "B");
    const int steps = (int)kv(meta, "steps"), n_rec = (int)kv(meta, "n_rec"), n_snap = (int)kv(meta, "n_snap");
    const double dt = kv(meta, "dt"), T = kv(meta, "T_ms");
    const int rec_every = (int)kv(meta, "rec_every"), w_every = (int)kv(meta, "w_every");
    const double stim_every = kv(meta, "stim_every") * dt, stim_len = kv(meta, "stim_len") * dt;
    const double warmup = kv(meta, "warmup") * dt, wall = kv(meta, "wall_s");
    const double version = kv(meta, "version");

    // ---- load ----
    std::vector<int> sp = read_bin<int>(P(folder, "spikes.bin"));
    const size_t ns = sp.size() / 2;
    std::vector<size_t> order(ns);
    std::iota(order.begin(), order.end(), (size_t)0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return sp[2 * a] < sp[2 * b]; });
    std::vector<double> t; std::vector<int> si;
    t.reserve(ns); si.reserve(ns);
    for (size_t k : order) {
        const double tt = sp[2 * k] * dt;
        if (tt >= warmup) { t.push_back(tt); si.push_back(sp[2 * k + 1]); }
    }
    std::vector<float> trs = read_bin<float>(P(folder, "trace_soma.bin"));
    std::vector<float> trb = read_bin<float>(P(folder, "trace_branch.bin"));
    std::vector<float> wm  = read_bin<float>(P(folder, "weights_mean.bin"));
    if ((int)trs.size() < n_rec * NT || (int)trb.size() < n_rec * NT * B || (int)wm.size() < 2 * n_snap) {
        std::printf("Output files are incomplete - run a simulation again\n"); return;
    }

    // ---- statistics ----
    const double dur = (T - warmup) / 1000.0;
    double cA = 0, cE = 0, cI = 0;
    for (int n : si) (n < N_A ? cA : n < N_E ? cE : cI) += 1;
    const double rate_A = cA / N_A / dur, rate_E = cE / (N_E - N_A) / dur, rate_I = cI / (N - N_E) / dur;

    // ISI CV per neuron (neurons with >= 5 spikes)
    std::vector<double> last(N, -1.0), s1(N, 0.0), s2(N, 0.0); std::vector<int> cnt(N, 0);
    for (size_t k = 0; k < t.size(); ++k) {          // t is sorted by time
        const int n = si[k];
        if (last[n] >= 0) { const double isi = t[k] - last[n]; s1[n] += isi; s2[n] += isi * isi; cnt[n]++; }
        last[n] = t[k];
    }
    double cv_sum = 0; int cv_n = 0;
    for (int n = 0; n < N; ++n)
        if (cnt[n] >= 4) {
            const double m = s1[n] / cnt[n], sd = std::sqrt(std::max(s2[n] / cnt[n] - m * m, 0.0));
            cv_sum += sd / m; ++cv_n;
        }
    const double cv = cv_n ? cv_sum / cv_n : NAN;

    std::vector<double> w_t(n_snap), w_A(n_snap), w_O(n_snap);
    for (int k = 0; k < n_snap; ++k) { w_t[k] = k * w_every * dt / 1000.0; w_A[k] = wm[2 * k]; w_O[k] = wm[2 * k + 1]; }
    w_t[n_snap - 1] = T / 1000.0;

    std::printf("Firing rate  group A: %.2f Hz   other E: %.2f Hz   I: %.2f Hz\n", rate_A, rate_E, rate_I);
    std::printf("ISI CV: %.2f\n", cv);
    std::printf("E->E weights inside group A: %.3f -> %.3f nS (%+.0f%%)\n", w_A[0], w_A.back(), 100 * (w_A.back() / w_A[0] - 1));
    std::printf("E->E weights elsewhere:      %.3f -> %.3f nS (%+.0f%%)\n", w_O[0], w_O.back(), 100 * (w_O.back() / w_O[0] - 1));
    std::printf("Simulation wall time: %.3f s (%.4f s per simulated second)\n", wall, wall / (T / 1000));

    // ---- synapse lists (plastic E->E weights at start and end) ----
    std::vector<float> w0p, w1p;
    if ((int)kv(meta, "full_weights", 1) == 1) {
        auto ptr = read_bin<int>(P(folder, "csr_ptr.bin")), eend = read_bin<int>(P(folder, "csr_eend.bin"));
        auto w0 = read_bin<float>(P(folder, "csr_w_init.bin")), w1 = read_bin<float>(P(folder, "csr_w_final.bin"));
        if ((int)ptr.size() == N + 1 && (int)eend.size() == N && !w0.empty() && w0.size() == w1.size())
            for (int post = 0; post < N_E; ++post)
                for (int q = ptr[post]; q < eend[post]; ++q) { w0p.push_back(w0[q]); w1p.push_back(w1[q]); }
    }

    // ---- figure data ----
    const int stride = std::max(1, N / 20000);      // huge networks: plot every k-th neuron
    std::vector<double> rx[3], ry[3];
    for (size_t k = 0; k < t.size(); ++k) {
        if (t[k] < T - 3000 || si[k] % stride) continue;
        const int g = si[k] < N_A ? 0 : si[k] < N_E ? 1 : 2;
        rx[g].push_back(t[k]); ry[g].push_back(si[k]);
    }
    std::vector<double> trt, vs, vb[4];
    for (int r = 0; r < n_rec; ++r) {
        const double tt = (double)r * rec_every * dt;
        if (tt < T - 1000) continue;
        trt.push_back(tt);
        vs.push_back(trs[(size_t)r * NT]);                         // neuron 0
        for (int k = 0; k < B && k < 4; ++k) vb[k].push_back(trb[((size_t)r * NT) * B + k]);
    }
    double hmax = 0;
    for (float w : w0p) hmax = std::max(hmax, (double)w);
    for (float w : w1p) hmax = std::max(hmax, (double)w);

    char head[256], lA[96], lE[96], lI[96], rtitle[160];
    std::snprintf(head, sizeof(head), "%d neurons: %.1f s simulated in %.2f s on the GPU (%.1fx real time, %.2f us/step)",
                  N, T / 1000, wall, T / 1000 / wall, 1e6 * wall / steps);
    std::snprintf(lA, sizeof(lA), "group A, stimulated (%.1f Hz)", rate_A);
    std::snprintf(lE, sizeof(lE), "other excitatory (%.1f Hz)", rate_E);
    std::snprintf(lI, sizeof(lI), "inhibitory (%.1f Hz)", rate_I);
    std::snprintf(rtitle, sizeof(rtitle), "%s%d neurons, last 3 s of activity (C++/CUDA v%g)   CV %.2f##raster",
                  stride == 1 ? "" : ("every " + std::to_string(stride) + "th neuron of ").c_str(), N, version, cv);
    const ImVec4 bcol[4] = { ImVec4(1.0f, 0.5f, 0.05f, 1), ImVec4(0.74f, 0.74f, 0.13f, 1),
                             ImVec4(0.09f, 0.75f, 0.81f, 1), ImVec4(0.89f, 0.47f, 0.76f, 1) };

    auto draw = [&]() {
        ImGui::TextUnformatted(head);
        const float H = plot_avail_h(), W = plot_avail_w();
        // raster
        if (ImPlot::BeginPlot(rtitle, ImVec2(-1, H * 0.34f))) {
            ImPlot::SetupAxes(nullptr, "neuron #");
            ImPlot::SetupAxesLimits(std::max(T - 3000, 0.0), T, 0, N, ImPlotCond_Always);
            ImPlot::SetupLegend(ImPlotLocation_NorthEast);
            for (double s = std::max(T - 3000, 0.0); s < T; s += stim_every)
                band("##stim", s, s + stim_len, 0, N, C_GREEN, 0.12f);
            ImPlot::PlotScatter(lA, rx[0].data(), ry[0].data(), (int)rx[0].size(), dot_spec(C_GREEN, 1.2f));
            ImPlot::PlotScatter(lE, rx[1].data(), ry[1].data(), (int)rx[1].size(), dot_spec(C_BLUE, 1.2f));
            ImPlot::PlotScatter(lI, rx[2].data(), ry[2].data(), (int)rx[2].size(), dot_spec(C_RED, 1.2f));
            ImPlot::EndPlot();
        }
        // branches of neuron 0
        if (ImPlot::BeginPlot("Neuron 0's dendritic branches, last 1 s", ImVec2(-1, H * 0.22f))) {
            ImPlot::SetupAxes(nullptr, "branch voltage (mV)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupLegend(ImPlotLocation_NorthEast);
            for (double s = std::max(T - 1000, 0.0); s < T; s += stim_every) band("##stim", s, s + stim_len, -90, 40, C_GREEN, 0.12f);
            for (int k = 0; k < B && k < 4; ++k) {
                char nm[48]; std::snprintf(nm, sizeof(nm), k == 0 ? "branch %d (gets stimulus)" : "branch %d", k);
                ImPlot::PlotLine(nm, trt.data(), vb[k].data(), (int)trt.size(), line_spec(bcol[k], k == 0 ? 1.6f : 0.8f));
            }
            ImPlot::EndPlot();
        }
        // soma of neuron 0
        if (ImPlot::BeginPlot("##soma", ImVec2(-1, H * 0.17f))) {
            ImPlot::SetupAxes("time (ms)", "soma voltage (mV)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            for (double s = std::max(T - 1000, 0.0); s < T; s += stim_every) band("##stim", s, s + stim_len, -90, 40, C_GREEN, 0.12f);
            ImPlot::PlotLine("neuron 0 soma", trt.data(), vs.data(), (int)trt.size(), line_spec(C_WHITE, 1.0f));
            ImPlot::EndPlot();
        }
        // weights over time + weight distribution
        const float hb = plot_avail_h();
        if (ImPlot::BeginPlot("STDP: learning over time", ImVec2(W * 0.5f - 4, hb))) {
            ImPlot::SetupAxes("time (s)", "mean E->E weight (nS)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("inside group A", w_t.data(), w_A.data(), n_snap, line_spec(C_GREEN, 2.0f));
            ImPlot::PlotLine("all other E->E", w_t.data(), w_O.data(), n_snap, line_spec(C_GRAY, 2.0f));
            ImPlot::EndPlot();
        }
        ImGui::SameLine();
        if (ImPlot::BeginPlot("Weight distribution", ImVec2(-1, hb))) {
            ImPlot::SetupAxes("E->E weight (nS)", "synapses", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            if (!w0p.empty()) {
                ImPlot::PlotHistogram("start", w0p.data(), (int)w0p.size(), 60, 1.0, ImPlotRange(0, hmax), fill_spec(C_GRAY, 0.5f));
                ImPlot::PlotHistogram("end", w1p.data(), (int)w1p.size(), 60, 1.0, ImPlotRange(0, hmax), fill_spec(C_BLUE, 0.5f));
            } else {
                ImPlot::PlotText("synapse lists not saved for N > 20,000", 0.5, 0.5);
            }
            ImPlot::EndPlot();
        }
    };
    if (show_window) plot_window("Brain - run results", 1600, 1000, draw, "test.png");
}

// =============================================================================
//  Sequence task
// =============================================================================
void analyze_seq(const std::string& folder, bool show_window) {
    const KV meta = read_kv(P(folder, "meta.txt"));
    if ((int)kv(meta, "seq", 0) != 1) { std::printf("The latest output is not a sequence-task run.\n"); return; }
    const double dt = kv(meta, "dt");
    const int steps = (int)kv(meta, "steps"), K = (int)kv(meta, "seq_K"), S = (int)kv(meta, "seq_size");
    const int cyc = (int)kv(meta, "seq_cycle"), slen = (int)kv(meta, "seq_len");
    const int tr0 = (int)kv(meta, "train0"), tr1 = (int)kv(meta, "train1"), te0 = (int)kv(meta, "test0");
    const int n_snap = (int)kv(meta, "n_snap"), w_every = (int)kv(meta, "w_every");
    const char* names = "ABCD";
    const ImVec4 colors[4] = { ImVec4(0.235f, 0.81f, 0.35f, 1), ImVec4(0.91f, 0.76f, 0.12f, 1),
                               ImVec4(0.88f, 0.31f, 0.71f, 1), ImVec4(0.17f, 0.78f, 0.90f, 1) };

    // spikes of the task groups, sorted by time
    std::vector<int> sp = read_bin<int>(P(folder, "spikes.bin"));
    std::vector<std::pair<int, int>> ev;              // (step, group)
    for (size_t k = 0; k + 1 < sp.size(); k += 2)
        if (sp[k + 1] < K * S) ev.push_back({ sp[k], sp[k + 1] / S });
    std::sort(ev.begin(), ev.end());

    // cue onsets in the test windows: cue A in even cycles, D in odd cycles
    auto cues = [&](int lo, int hi, int which) {
        std::vector<int> out;
        for (int c0 = (lo + cyc - 1) / cyc * cyc; c0 < hi - cyc; c0 += cyc) {
            if (c0 < te0) continue;
            if (((c0 / cyc) & 1) == (which == 0 ? 0 : 1)) out.push_back(c0);
        }
        return out;
    };
    const auto before_A = cues(0, tr0, 0), before_D = cues(0, tr0, K - 1);
    const auto after_A = cues(tr1, steps, 0), after_D = cues(tr1, steps, K - 1);

    const int win = (int)std::lround(100 / dt), pre = (int)std::lround(20 / dt), binw = (int)std::lround(2 / dt);
    const int nb = (win + pre) / binw;               // bins from -pre to win
    std::vector<double> tcent(nb);
    for (int b = 0; b < nb; ++b) tcent[b] = (-pre + (b + 0.5) * binw) * dt;

    // firing rate (Hz per neuron) of each group around the cue onsets
    auto psth = [&](const std::vector<int>& onsets) {
        std::vector<std::vector<double>> out(K, std::vector<double>(nb, 0.0));
        if (onsets.empty()) return out;
        for (int o : onsets) {
            auto it = std::lower_bound(ev.begin(), ev.end(), std::make_pair(o - pre, -1));
            for (; it != ev.end() && it->first < o + win; ++it) {    // [o - pre, o + win), as brain.py
                const int b = (it->first - (o - pre)) / binw;
                if (b >= 0 && b < nb) out[it->second][b] += 1;
            }
        }
        const double norm = onsets.size() * (double)S * binw * dt / 1000.0;
        for (auto& g : out) for (double& v : g) v /= norm;
        return out;
    };
    const auto bA = psth(before_A), aA = psth(after_A), bD = psth(before_D), aD = psth(after_D);
    auto resp = [&](const std::vector<std::vector<double>>& r, int g) {
        double s = 0; int n = 0;
        for (int b = 0; b < nb; ++b) if (tcent[b] >= 5 && tcent[b] <= 80) { s += r[g][b]; ++n; }
        return n ? s / n : 0.0;
    };
    auto peak = [&](const std::vector<std::vector<double>>& r, int g) {
        double best = -1, at = 0;
        for (int b = 0; b < nb; ++b) if (tcent[b] >= 2 && tcent[b] <= 90 && r[g][b] > best) { best = r[g][b]; at = tcent[b]; }
        return at;
    };

    std::printf("\n================ SEQUENCE TASK ================\n");
    std::printf("cues: %zu x A and %zu x D before training, %zu x A and %zu x D after\n",
                before_A.size(), before_D.size(), after_A.size(), after_D.size());
    std::printf("Response to cue A (mean rate 5-80 ms after cue, Hz)    before   after    peak after (ms)\n");
    for (int g = 1; g < K; ++g)
        std::printf("   group %c                                          %6.1f  %6.1f    %5.0f\n",
                    names[g], resp(bA, g), resp(aA, g), peak(aA, g));
    std::printf("Control, response to cue D (should NOT run backwards)  before   after\n");
    for (int g = K - 2; g >= 0; --g)
        std::printf("   group %c                                          %6.1f  %6.1f\n", names[g], resp(bD, g), resp(aD, g));
    double fwd = 0, rev = 0;
    for (int g = 1; g < K; ++g) fwd += resp(aA, g) - resp(bA, g);
    for (int g = 0; g < K - 1; ++g) rev += resp(aD, g) - resp(bD, g);
    fwd /= (K - 1); rev /= (K - 1);
    bool ordered = true;
    for (int g = 1; g + 1 < K; ++g) ordered = ordered && peak(aA, g) < peak(aA, g + 1);
    std::printf("Replay gain (forward) %+.1f Hz | backward %+.1f Hz | peaks in order B<C<D: %s\n",
                fwd, rev, ordered ? "yes" : "no");
    const char* verdict = (fwd > 2.0 && fwd > 2 * std::max(rev, 0.5) && ordered) ? "REPLAY: cue A sets off B, C, D in order"
                        : fwd > 1.0 ? "PARTIAL: forward response grew, but not a clean ordered chain"
                        : "NO REPLAY yet: try more training or a higher learning rate";
    std::printf("VERDICT: %s\n================================================\n", verdict);

    // weights between groups over time: [snap][pre][post]
    std::vector<float> gw = read_bin<float>(P(folder, "group_weights.bin"));
    if ((int)gw.size() < n_snap * K * K) { std::printf("group_weights.bin missing\n"); return; }
    std::vector<double> wt(n_snap);
    for (int k = 0; k < n_snap; ++k) wt[k] = k * w_every * dt / 1000.0;
    wt[n_snap - 1] = steps * dt / 1000.0;
    std::vector<std::vector<double>> fwdw(K - 1, std::vector<double>(n_snap)), revw(K - 1, std::vector<double>(n_snap));
    for (int a = 0; a + 1 < K; ++a)
        for (int k = 0; k < n_snap; ++k) {
            fwdw[a][k] = gw[(size_t)k * K * K + a * K + a + 1];
            revw[a][k] = gw[(size_t)k * K * K + (a + 1) * K + a];
        }
    std::vector<double> final_w(K * K);
    double fmax = 0;
    for (int i = 0; i < K * K; ++i) { final_w[i] = gw[(size_t)(n_snap - 1) * K * K + i]; fmax = std::max(fmax, final_w[i]); }

    // shared y range for the four response plots, and for the weight plot
    double rmax = 1.0, wlo = 1e9, whi = -1e9;
    for (auto* r : { &bA, &aA, &bD, &aD }) for (auto& g : *r) for (double v : g) rmax = std::max(rmax, v);
    rmax *= 1.1;
    for (int a = 0; a + 1 < K; ++a) for (int k = 0; k < n_snap; ++k) {
        wlo = std::min(wlo, std::min(fwdw[a][k], revw[a][k])); whi = std::max(whi, std::max(fwdw[a][k], revw[a][k]));
    }
    const double wpad = std::max(0.02 * whi, (whi - wlo) * 0.1);
    wlo -= wpad; whi += wpad;

    char head[320];
    std::snprintf(head, sizeof(head), "Sequence task: %d neurons, %.0f s training, learning x%g, %.0f s simulated in %.1f s  |  %s",
                  (int)kv(meta, "N"), (tr1 - tr0) * dt / 1000, kv(meta, "lr", 1), steps * dt / 1000, kv(meta, "wall_s"), verdict);

    auto draw = [&]() {
        ImGui::TextUnformatted(head);
        const float H = plot_avail_h(), W = plot_avail_w();
        const char* titles[4] = { "Cue A alone: before training", "Cue D alone (control): before training",
                                  "Cue A alone: after training", "Cue D alone (control): after training" };
        const std::vector<std::vector<double>>* rs[4] = { &bA, &bD, &aA, &aD };
        const char keys[4] = { 'A', 'D', 'A', 'D' };
        for (int i = 0; i < 4; ++i) {
            if (i % 2) ImGui::SameLine();
            if (ImPlot::BeginPlot(titles[i], ImVec2(i % 2 ? -1 : W * 0.5f - 4, H * 0.31f))) {
                ImPlot::SetupAxes(i >= 2 ? "time after cue (ms)" : nullptr, "rate (Hz)", ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, rmax, ImPlotCond_Once);
                ImPlot::SetupLegend(ImPlotLocation_NorthEast);
                band("##cue", 0, slen * dt, 0, rmax, C_GRAY, 0.15f);
                for (int g = 0; g < K && g < 4; ++g) {
                    char nm[24]; std::snprintf(nm, sizeof(nm), "group %c", names[g]);
                    const bool cued = names[g] == keys[i];
                    ImVec4 c = colors[g]; if (cued) c.w = 0.5f;
                    ImPlot::PlotLine(nm, tcent.data(), (*rs[i])[g].data(), nb, line_spec(c, cued ? 1.0f : 2.0f));
                }
                ImPlot::EndPlot();
            }
        }
        const float hb = plot_avail_h();
        if (ImPlot::BeginPlot("Weights between groups", ImVec2(W * 0.5f - 4, hb))) {
            ImPlot::SetupAxes("time (s)", "mean weight (nS)", ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisLimits(ImAxis_Y1, wlo, whi, ImPlotCond_Once);
            ImPlot::SetupLegend(ImPlotLocation_NorthWest);
            band("##train", tr0 * dt / 1000, tr1 * dt / 1000, wlo, whi, C_GRAY, 0.10f);
            for (int a = 0; a + 1 < K; ++a) {
                char f[32], r[32];
                std::snprintf(f, sizeof(f), "%c->%c (forward)", names[a], names[a + 1]);
                std::snprintf(r, sizeof(r), "%c->%c (reverse)", names[a + 1], names[a]);
                ImPlot::PlotLine(f, wt.data(), fwdw[a].data(), n_snap, line_spec(colors[a + 1], 2.0f));
                ImVec4 c = colors[a]; c.w = 0.6f;
                ImPlot::PlotLine(r, wt.data(), revw[a].data(), n_snap, line_spec(c, 1.0f));
            }
            ImPlot::EndPlot();
        }
        ImGui::SameLine();
        if (ImPlot::BeginPlot("Final mean weights (nS): row = from, column = to", ImVec2(-1, hb), ImPlotFlags_NoLegend)) {
            static const char* labels[4] = { "A", "B", "C", "D" };
            const double pos[4] = { 0.125, 0.375, 0.625, 0.875 };
            ImPlot::SetupAxes("to group", "from group", ImPlotAxisFlags_NoGridLines, ImPlotAxisFlags_NoGridLines);
            ImPlot::SetupAxisTicks(ImAxis_X1, pos, K, labels);
            const double ypos[4] = { 0.875, 0.625, 0.375, 0.125 };   // row 0 (A) at the top
            ImPlot::SetupAxisTicks(ImAxis_Y1, ypos, K, labels);
            ImPlot::PushColormap(ImPlotColormap_Plasma);
            ImPlot::PlotHeatmap("##w", final_w.data(), K, K, 0.0, fmax, "%.2f");
            ImPlot::PopColormap();
            ImPlot::EndPlot();
        }
    };
    if (show_window) plot_window("Brain - sequence task", 1600, 1000, draw, "sequence.png");
}

// =============================================================================
//  Frequency task
// =============================================================================
void analyze_tone(const std::string& folder, bool show_window) {
    const KV meta = read_kv(P(folder, "meta.txt"));
    if ((int)kv(meta, "tone", 0) != 1) { std::printf("The latest output is not a frequency-task run.\n"); return; }
    const double dt = kv(meta, "dt"), target = kv(meta, "tone_hz");
    const int N = (int)kv(meta, "N"), N_E = (int)kv(meta, "N_E");
    const int tone_steps = (int)kv(meta, "tone_steps"), K = (int)kv(meta, "tone_tests");
    const int NB = 64, NT = 48, TGT = 4;                 // test index 4 = the target tone
    auto tr = read_bin<float>(P(folder, "tone_trials.bin"));
    auto reg = read_bin<uint8_t>(P(folder, "region.bin"));
    auto band = read_bin<int16_t>(P(folder, "band.bin"));
    auto tw = read_bin<float>(P(folder, "tone_weights.bin"));
    auto sp = read_bin<int>(P(folder, "spikes.bin"));
    if ((int)reg.size() != N || (int)band.size() != N || tr.size() % 5 || tw.size() != 256) {
        std::printf("Frequency-task files are missing or incomplete.\n"); return;
    }
    // neuron groups: A1 excitatory cells by pitch band, mouth cells by tone
    std::vector<int> grp(N, -1);                         // 0..63 A1 band, 64..111 mouth tone
    std::vector<double> nG(NB + NT, 0.0);
    for (int n = 0; n < N; ++n) {
        if (reg[n] == 2 && n < N_E && band[n] >= 0) grp[n] = band[n];
        else if (reg[n] == 3 && band[n] >= 0) grp[n] = NB + band[n];
        if (grp[n] >= 0) nG[grp[n]] += 1;
    }
    std::vector<std::pair<int, int>> ev;                 // (step, group) for spikes of those cells
    for (size_t k = 0; k + 1 < sp.size(); k += 2) { const int g = grp[sp[k + 1]]; if (g >= 0) ev.push_back({ sp[k], g }); }
    std::sort(ev.begin(), ev.end());
    auto count = [&](int a, int b, std::vector<double>& out) {   // spikes per group in steps [a, b)
        std::fill(out.begin(), out.end(), 0.0);
        auto it = std::lower_bound(ev.begin(), ev.end(), std::make_pair(a, -1));
        for (; it != ev.end() && it->first < b; ++it) out[it->second] += 1;
    };
    // response = rate during the tone minus rate in the 150 ms before it (Hz per neuron)
    const int ntr = (int)(tr.size() / 5), base_steps = (int)std::lround(150 / dt);
    std::vector<double> resp(2 * K * (NB + NT), 0.0), reps(2 * K, 0.0), on(NB + NT), off(NB + NT);
    for (int i = 0; i < ntr; ++i) {
        const int start = (int)tr[5 * i], phase = (int)tr[5 * i + 1], k = (int)tr[5 * i + 4];
        if (phase == 1) continue;
        const int ph = phase == 0 ? 0 : 1;
        count(start, start + tone_steps, on);
        count(start - base_steps, start, off);
        for (int g = 0; g < NB + NT; ++g) {
            if (nG[g] <= 0) continue;
            const double r_on = on[g] / (nG[g] * tone_steps * dt / 1000.0), r_off = off[g] / (nG[g] * base_steps * dt / 1000.0);
            resp[((size_t)ph * K + k) * (NB + NT) + g] += r_on - r_off;
        }
        reps[ph * K + k] += 1;
    }
    for (int ph = 0; ph < 2; ++ph) for (int k = 0; k < K; ++k)
        for (int g = 0; g < NB + NT; ++g) if (reps[ph * K + k] > 0) resp[((size_t)ph * K + k) * (NB + NT) + g] /= reps[ph * K + k];
    auto R = [&](int ph, int k, int g) { return resp[((size_t)ph * K + k) * (NB + NT) + g]; };

    // response of the cells tuned to each test tone (its own band +-1)
    std::vector<double> hz(K), a1(2 * K, 0.0), mo(2 * K, 0.0);
    for (int k = 0; k < K; ++k) {
        hz[k] = target * std::pow(2.0, (k - 4) * 0.5);
        const double cb = NB * std::log(hz[k] / 20.0) / std::log(800.0) - 0.5;
        const double ct = NT * std::log(hz[k] / 20.0) / std::log(800.0) - 0.5;
        for (int ph = 0; ph < 2; ++ph) {
            double s = 0; int c = 0;
            for (int b = (int)std::lround(cb) - 1; b <= (int)std::lround(cb) + 1; ++b) if (b >= 0 && b < NB && nG[b] > 0) { s += R(ph, k, b); ++c; }
            a1[ph * K + k] = c ? s / c : 0.0;
            s = 0; c = 0;
            for (int t = (int)std::lround(ct) - 1; t <= (int)std::lround(ct) + 1; ++t) if (t >= 0 && t < NT && nG[NB + t] > 0) { s += R(ph, k, NB + t); ++c; }
            mo[ph * K + k] = c ? s / c : 0.0;
        }
    }
    double oth_a = 0, oth_m = 0;
    for (int k = 0; k < K; ++k) if (k != TGT) { oth_a += a1[K + k] - a1[k]; oth_m += mo[K + k] - mo[k]; }
    oth_a /= (K - 1); oth_m /= (K - 1);
    const double gt_a = a1[K + TGT] - a1[TGT], gt_m = mo[K + TGT] - mo[TGT];

    std::printf("\n================ FREQUENCY TASK ================\n");
    std::printf("Trained tone %.0f Hz, %d training presentations, reward strength %g\n",
                target, (int)kv(meta, "tone_train"), kv(meta, "reward"));
    std::printf("Response of the cells tuned to each tone (Hz above baseline)\n");
    std::printf("   tone (Hz)     A1 before  A1 after   change    mouth before  mouth after\n");
    for (int k = 0; k < K; ++k)
        std::printf("   %7.0f%s   %8.2f  %8.2f  %+8.2f      %8.2f    %8.2f\n", hz[k], k == TGT ? " *" : "  ",
                    a1[k], a1[K + k], a1[K + k] - a1[k], mo[k], mo[K + k]);
    std::printf("A1: trained tone changed %+.2f Hz, other tones %+.2f Hz on average\n", gt_a, oth_a);
    std::printf("Mouth: trained tone changed %+.2f Hz, other tones %+.2f Hz on average\n", gt_m, oth_m);
    const double tb = NB * std::log(target / 20.0) / std::log(800.0) - 0.5;
    const int tbi = std::max(0, std::min(NB - 1, (int)std::lround(tb)));
    std::printf("Nerve -> A1 weight at the trained band: %.3f -> %.3f nS (all bands: %.3f -> %.3f)\n",
                tw[tbi], tw[64 + tbi], [&] { double s = 0; for (int b = 0; b < NB; ++b) s += tw[b]; return s / NB; }(),
                [&] { double s = 0; for (int b = 0; b < NB; ++b) s += tw[64 + b]; return s / NB; }());
    const char* verdict = (gt_a > 0.5 && gt_a > 2.0 * std::max(std::fabs(oth_a), 0.25)) ? "SELECTIVE: the trained tone's response grew more than the others"
                        : (std::fabs(gt_a) > 0.5 || std::fabs(oth_a) > 0.5) ? "GENERAL: responses changed, but not specifically for the trained tone"
                        : "NO CLEAR CHANGE: try more training or a higher reward strength";
    std::printf("VERDICT: %s\n================================================\n", verdict);

    // ---- figure ----
    std::vector<double> mapB((size_t)K * NB), mapA((size_t)K * NB);
    double vmin = 0, vmax = 1e-3;
    for (int k = 0; k < K; ++k) for (int b = 0; b < NB; ++b) {
        mapB[(size_t)k * NB + b] = R(0, k, b); mapA[(size_t)k * NB + b] = R(1, k, b);
        vmin = std::min(vmin, std::min(R(0, k, b), R(1, k, b))); vmax = std::max(vmax, std::max(R(0, k, b), R(1, k, b)));
    }
    std::vector<std::string> lab(K); std::vector<const char*> labp(K); std::vector<double> ypos(K), xk(K);
    for (int k = 0; k < K; ++k) {
        char b[32]; std::snprintf(b, sizeof(b), hz[k] >= 1000 ? "%.1fk" : "%.0f", hz[k] >= 1000 ? hz[k] / 1000 : hz[k]);
        lab[k] = b; labp[k] = lab[k].c_str(); ypos[k] = K - k - 0.5; xk[k] = k;
    }
    std::vector<double> bx(NB), wA0(NB), wA1(NB), mx(NT), wM0(NT), wM1(NT);
    for (int b = 0; b < NB; ++b) { bx[b] = b; wA0[b] = tw[b]; wA1[b] = tw[64 + b]; }
    for (int t = 0; t < NT; ++t) { mx[t] = (t + 0.5) * NB / NT; wM0[t] = tw[128 + t]; wM1[t] = tw[192 + t]; }
    char head[320];
    std::snprintf(head, sizeof(head), "Frequency task: %d neurons, trained %.0f Hz with dopamine (%d presentations, strength %g), %.0f s simulated in %.1f s  |  %s",
                  N, target, (int)kv(meta, "tone_train"), kv(meta, "reward"), kv(meta, "steps") * dt / 1000, kv(meta, "wall_s"), verdict);
    const double tgt_x = TGT, tgt_b = tb;
    auto draw = [&]() {
        ImGui::TextUnformatted(head);
        const float H = plot_avail_h(), W = plot_avail_w();
        const char* titles[2] = { "A1 response before training (Hz above baseline)", "A1 response after training" };
        const std::vector<double>* maps[2] = { &mapB, &mapA };
        for (int i = 0; i < 2; ++i) {
            if (i) ImGui::SameLine();
            if (ImPlot::BeginPlot(titles[i], ImVec2(i ? -1 : W * 0.5f - 4, H * 0.48f), ImPlotFlags_NoLegend)) {
                ImPlot::SetupAxes("A1 pitch band (low -> high)", "test tone (Hz)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisTicks(ImAxis_Y1, ypos.data(), K, labp.data());
                ImPlot::PushColormap(ImPlotColormap_Viridis);
                ImPlot::PlotHeatmap("##map", maps[i]->data(), K, NB, vmin, vmax, nullptr, ImPlotPoint(0, 0), ImPlotPoint(NB, K));
                ImPlot::PopColormap();
                ImPlot::EndPlot();
            }
        }
        const float hb = plot_avail_h(), w3 = W / 3 - 6;
        if (ImPlot::BeginPlot("A1 cells tuned to each tone", ImVec2(w3, hb))) {
            ImPlot::SetupAxes("test tone (Hz)", "response (Hz)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisTicks(ImAxis_X1, xk.data(), K, labp.data());
            ImPlot::PlotInfLines("trained tone", &tgt_x, 1, line_spec(ImVec4(0.3f, 1.0f, 0.5f, 0.6f), 2.0f));
            ImPlotSpec b = line_spec(C_GRAY, 2.0f); b.Marker = ImPlotMarker_Circle;
            ImPlotSpec a = line_spec(ImVec4(1.0f, 0.6f, 0.2f, 1), 2.0f); a.Marker = ImPlotMarker_Circle;
            ImPlot::PlotLine("before", xk.data(), &a1[0], K, b);
            ImPlot::PlotLine("after", xk.data(), &a1[K], K, a);
            ImPlot::EndPlot();
        }
        ImGui::SameLine();
        if (ImPlot::BeginPlot("Mouth cells tuned to each tone", ImVec2(w3, hb))) {
            ImPlot::SetupAxes("test tone (Hz)", "response (Hz)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisTicks(ImAxis_X1, xk.data(), K, labp.data());
            ImPlot::PlotInfLines("trained tone", &tgt_x, 1, line_spec(ImVec4(0.3f, 1.0f, 0.5f, 0.6f), 2.0f));
            ImPlotSpec b = line_spec(C_GRAY, 2.0f); b.Marker = ImPlotMarker_Circle;
            ImPlotSpec a = line_spec(ImVec4(0.9f, 0.4f, 0.9f, 1), 2.0f); a.Marker = ImPlotMarker_Circle;
            ImPlot::PlotLine("before", xk.data(), &mo[0], K, b);
            ImPlot::PlotLine("after", xk.data(), &mo[K], K, a);
            ImPlot::EndPlot();
        }
        ImGui::SameLine();
        if (ImPlot::BeginPlot("Mean synapse strength by pitch", ImVec2(-1, hb))) {
            ImPlot::SetupAxes("pitch band (low -> high)", "weight (nS)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotInfLines("trained tone", &tgt_b, 1, line_spec(ImVec4(0.3f, 1.0f, 0.5f, 0.6f), 2.0f));
            ImPlot::PlotLine("nerve->A1 before", bx.data(), wA0.data(), NB, line_spec(C_GRAY, 1.5f));
            ImPlot::PlotLine("nerve->A1 after", bx.data(), wA1.data(), NB, line_spec(ImVec4(1.0f, 0.6f, 0.2f, 1), 2.0f));
            ImPlot::PlotLine("A1->mouth before", mx.data(), wM0.data(), NT, line_spec(ImVec4(0.5f, 0.5f, 0.7f, 1), 1.5f));
            ImPlot::PlotLine("A1->mouth after", mx.data(), wM1.data(), NT, line_spec(ImVec4(0.9f, 0.4f, 0.9f, 1), 2.0f));
            ImPlot::EndPlot();
        }
    };
    if (show_window) plot_window("Brain - frequency task", 1600, 1000, draw, "frequency.png");
}

// =============================================================================
//  Walking creatures: distance, survival and dopamine over the run (walking.png)
// =============================================================================
void analyze_walk(const std::string& folder, bool show_window) {
    const KV meta = read_kv(P(folder, "meta.txt"));
    if ((int)kv(meta, "walk", 0) != 1) { std::printf("The latest output is not a walking-creatures run.\n"); return; }
    const int K = (int)kv(meta, "brains"), n = (int)kv(meta, "n");
    const bool octo = kv(meta, "modular", 0) != 0;
    const double sim_s = kv(meta, "sim_s"), age_s = kv(meta, "age_s", 0);
    const auto lives = read_bin<float>(P(folder, "walk_lives.bin"));      // brain, life, t, dist, died
    const auto dop   = read_bin<float>(P(folder, "walk_dopamine.bin"));   // t, then K values
    const auto ev    = read_bin<float>(P(folder, "walk_events.bin"));     // t, replaced, copied
    if (K < 1 || lives.size() % 5 || dop.size() % (size_t)(K + 1) || ev.size() % 3) {
        std::printf("Walking-creature files are missing or incomplete.\n"); return;
    }
    static const ImVec4 COL[8] = { ImVec4(1.00f, 0.63f, 0.16f, 1), ImVec4(0.24f, 0.78f, 1.00f, 1), ImVec4(0.47f, 0.90f, 0.35f, 1),
                                   ImVec4(0.94f, 0.35f, 0.78f, 1), ImVec4(0.98f, 0.90f, 0.27f, 1), ImVec4(0.59f, 0.51f, 1.00f, 1),
                                   ImVec4(1.00f, 0.39f, 0.39f, 1), ImVec4(0.78f, 0.78f, 0.78f, 1) };
    const int NL = (int)(lives.size() / 5);
    // per brain: life number, distance, rolling mean; survival per block of 10 lives
    std::vector<std::vector<double>> lx(K), ly(K), rm(K), sx(K), sy(K);
    std::vector<double> all_t, all_d, all_rm_t, all_rm;
    int deaths = 0, best_b = -1; double best_d = -1e9;
    for (int i = 0; i < NL; ++i) {
        const int b = (int)lives[5 * i]; if (b < 0 || b >= K) continue;
        lx[b].push_back(lives[5 * i + 1]); ly[b].push_back(lives[5 * i + 3]);
        all_t.push_back(lives[5 * i + 2] / 60.0); all_d.push_back(lives[5 * i + 3]);
        deaths += lives[5 * i + 4] > 0.5f;
        if (lives[5 * i + 3] > best_d) { best_d = lives[5 * i + 3]; best_b = b; }
    }
    const int W = std::max(5, std::min(25, NL / std::max(1, 10 * K)));     // rolling-mean window (lives)
    for (int b = 0; b < K; ++b) {
        const auto& y = ly[b];
        for (size_t i = 0; i < y.size(); ++i) {
            const size_t a = i + 1 >= (size_t)W ? i + 1 - W : 0;
            double s = 0; for (size_t j = a; j <= i; ++j) s += y[j];
            rm[b].push_back(s / (double)(i - a + 1));
        }
        int alive = 0, cnt = 0;                                           // % of lives survived, per 10 lives
        for (int i = 0; i < NL; ++i) {
            if ((int)lives[5 * i] != b) continue;
            alive += lives[5 * i + 4] < 0.5f; ++cnt;
            if (cnt == 10) { sx[b].push_back(lives[5 * i + 1]); sy[b].push_back(10.0 * alive); alive = 0; cnt = 0; }
        }
    }
    {   // population trend over time: rolling mean over the last W*K lives
        const int Wa = W * std::max(1, K);
        for (size_t i = 0; i < all_d.size(); ++i) {
            const size_t a = i + 1 >= (size_t)Wa ? i + 1 - Wa : 0;
            double s = 0; for (size_t j = a; j <= i; ++j) s += all_d[j];
            all_rm_t.push_back(all_t[i]); all_rm.push_back(s / (double)(i - a + 1));
        }
    }
    const int ND = (int)(dop.size() / (K + 1));
    std::vector<double> dt_min(ND); std::vector<std::vector<double>> dv(K, std::vector<double>(ND));
    for (int i = 0; i < ND; ++i) {
        dt_min[i] = dop[(size_t)i * (K + 1)] / 60.0;
        for (int b = 0; b < K; ++b) dv[b][i] = dop[(size_t)i * (K + 1) + 1 + b];
    }
    std::vector<double> ev_t; for (size_t i = 0; i + 2 < ev.size(); i += 3) ev_t.push_back(ev[i] / 60.0);
    // first vs last quarter of all lives: is the population getting better?
    double q1 = 0, q4 = 0; int nq = NL / 4;
    for (int i = 0; i < nq; ++i) { q1 += all_d[i]; q4 += all_d[NL - nq + i]; }
    if (nq > 0) { q1 /= nq; q4 /= nq; }
    char head[480];
    std::snprintf(head, sizeof(head),
        "Walking creatures: %d %s%s brains x %d neurons  |  %.1f min simulated (%.1f min total experience)  |  %d lives, %.0f%% died  |  "
        "best life %+.2f m (brain %d)  |  mean distance, first quarter of lives %+.2f m -> last quarter %+.2f m  |  "
        "%d generations, mutation %.1f%%",
        K, octo ? "octopus" : "blob", kv(meta, "cpg", 0) != 0 ? " + rhythm" : "", n, sim_s / 60.0, (age_s + sim_s) / 60.0, NL, NL ? 100.0 * deaths / NL : 0.0,
        best_d, best_b + 1, q1, q4, (int)kv(meta, "generations", 0), 100.0 * kv(meta, "mutation", 0));
    std::printf("%s\n", head);
    auto draw = [&]() {
        ImGui::TextWrapped("%s", head);
        const float H = plot_avail_h(), Wd = plot_avail_w();
        if (ImPlot::BeginPlot("Distance per life (m): dots = each life, lines = rolling mean", ImVec2(Wd * 0.5f - 4, H * 0.5f - 4))) {
            ImPlot::SetupAxes("life", "distance (m)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            for (int b = 0; b < K; ++b) {
                if (ly[b].empty()) continue;
                char nm[24]; std::snprintf(nm, sizeof(nm), "brain %d", b + 1);
                ImVec4 c = COL[b % 8]; ImVec4 cd = c; cd.w = 0.35f;
                ImPlot::PlotScatter(nm, lx[b].data(), ly[b].data(), (int)ly[b].size(), dot_spec(cd, 2.0f));
                ImPlot::PlotLine(nm, lx[b].data(), rm[b].data(), (int)rm[b].size(), line_spec(c, 2.0f));
            }
            ImPlot::EndPlot();
        }
        ImGui::SameLine();
        if (ImPlot::BeginPlot("Whole population over time (m)", ImVec2(-1, H * 0.5f - 4))) {
            ImPlot::SetupAxes("simulated time (min)", "distance (m)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            if (!ev_t.empty()) {
                ImPlotSpec es = line_spec(ImVec4(0.6f, 0.6f, 0.6f, 0.25f), 1.0f);
                ImPlot::PlotInfLines("child born (evolution)", ev_t.data(), (int)ev_t.size(), es);
            }
            ImPlot::PlotScatter("each life", all_t.data(), all_d.data(), (int)all_d.size(), dot_spec(ImVec4(0.8f, 0.8f, 0.8f, 0.35f), 2.0f));
            ImPlot::PlotLine("rolling mean", all_rm_t.data(), all_rm.data(), (int)all_rm.size(), line_spec(ImVec4(0.3f, 1.0f, 0.5f, 1), 2.5f));
            ImPlot::EndPlot();
        }
        const float hb = plot_avail_h();
        if (ImPlot::BeginPlot("Survival: % of lives that lasted the full life length (per 10 lives)", ImVec2(Wd * 0.5f - 4, hb))) {
            ImPlot::SetupAxes("life", "survived (%)", ImPlotAxisFlags_AutoFit, 0);
            ImPlot::SetupAxisLimits(ImAxis_Y1, -5, 105, ImPlotCond_Always);
            for (int b = 0; b < K; ++b) {
                if (sy[b].empty()) continue;
                char nm[24]; std::snprintf(nm, sizeof(nm), "brain %d", b + 1);
                ImPlotSpec ls = line_spec(COL[b % 8], 2.0f); ls.Marker = ImPlotMarker_Circle; ls.MarkerSize = 3;
                ImPlot::PlotLine(nm, sx[b].data(), sy[b].data(), (int)sy[b].size(), ls);
            }
            ImPlot::EndPlot();
        }
        ImGui::SameLine();
        if (ImPlot::BeginPlot("Dopamine per second (smoothed)", ImVec2(-1, hb))) {
            ImPlot::SetupAxes("simulated time (min)", "dopamine / s", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            const double zero = 0.0;
            ImPlot::PlotInfLines("##zero", &zero, 1, [] { ImPlotSpec z = line_spec(ImVec4(0.6f, 0.6f, 0.6f, 0.5f), 1.0f); z.Flags = ImPlotInfLinesFlags_Horizontal; return z; }());
            for (int b = 0; b < K; ++b) {
                char nm[24]; std::snprintf(nm, sizeof(nm), "brain %d", b + 1);
                if (ND) ImPlot::PlotLine(nm, dt_min.data(), dv[b].data(), ND, line_spec(COL[b % 8], 1.5f));
            }
            ImPlot::EndPlot();
        }
    };
    if (show_window) plot_window("Brain - walking creatures", 1600, 1000, draw, "walking.png");
}

void plot_latest(const std::string& folder) {
    if (!file_exists(P(folder, "meta.txt"))) { std::printf("No output yet - run a simulation first\n"); return; }
    const KV m = read_kv(P(folder, "meta.txt"));
    if ((int)kv(m, "walk", 0) == 1) analyze_walk(folder);
    else if ((int)kv(m, "tone", 0) == 1) analyze_tone(folder);
    else if ((int)kv(m, "seq", 0) == 1) analyze_seq(folder);
    else plot_run(folder);
}
