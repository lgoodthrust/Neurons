// app.cpp - the program's entry point: the menu (replaces brain.py and brain.bat).
//
//   Brain.exe                                  menu
//   Brain.exe rate T_ms nmda mode N [extra]    old command line (mode: run | validate | live | seq)
#include "sim.h"
#include "results.h"
#include "validate.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "version.h"
#define BRAIN_VERSION_STR BRAIN_VERSION

static const std::string HERE = "";            // output folder: the exe's own folder (set in main)

// Output files, PNGs and bench_log.txt all go next to Brain.exe, wherever it is started from.
static void go_to_exe_folder() {
#ifdef _WIN32
    char path[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        std::string dir(path, n);
        const size_t cut = dir.find_last_of("\\/");
        if (cut != std::string::npos) SetCurrentDirectoryA(dir.substr(0, cut).c_str());
    }
#endif
}

template <class T> static T ask(const char* prompt, T fallback) {
    std::cout << "  " << prompt << " [" << fallback << "]: " << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) return fallback;
    const size_t a = line.find_first_not_of(" \t");
    if (a == std::string::npos) return fallback;
    try {
        if (std::is_integral<T>::value) return (T)std::stoll(line.substr(a));
        return (T)std::stod(line.substr(a));
    } catch (...) {
        std::cout << "  (not a number, using " << fallback << ")\n";
        return fallback;
    }
}

// ---------------------------------------------------------------------------
//  Brain options (menu 13): used by every run, kept in brain_options.txt next to Brain.exe
// ---------------------------------------------------------------------------
struct BrainOptions { int syn = 160; float dt = 0.1f; };
static BrainOptions g_opt;
static const char* kOptFile = "brain_options.txt";

static void load_options() {
    std::ifstream f(kOptFile);
    std::string key;
    double v;
    while (f >> key >> v) {
        if (key == "synapses_per_neuron") g_opt.syn = std::max(20, std::min(1000, (int)v));
        else if (key == "dt_ms") g_opt.dt = snap_dt((float)v);
    }
}
static void save_options() {
    std::ofstream f(kOptFile, std::ios::trunc);
    f << "synapses_per_neuron " << g_opt.syn << "\ndt_ms " << g_opt.dt << "\n";
}

static RunConfig cfg(const char* mode, int N, float T_ms) {
    RunConfig c; c.mode = mode; c.N = N; c.T_ms = T_ms;
    c.syn = g_opt.syn; c.dt = g_opt.dt;
    return c;
}

static void cmd_options() {
    std::printf("\n  Brain options (used by every run from now on; saved brains keep their own)\n");
    std::printf("  Synapses per neuron: how many inputs come from simulated neurons. The rest of a\n"
                "  neuron's ~6,900 inputs arrive as background, so the total stays the same.\n");
    g_opt.syn = std::max(20, std::min(1000, (int)ask("synapses per neuron (20 - 1,000; default 160)", g_opt.syn)));
    std::printf("  Time step: smaller = more accurate and slower. It must divide 1 ms evenly:\n"
                "  0.0625, 0.0667, 0.0714, 0.0769, 0.0833, 0.0909, 0.1, 0.111, 0.125, 0.143, 0.167, 0.2, 0.25, 0.333, 0.5\n");
    const float want = ask("time step in ms (default 0.1)", g_opt.dt);
    g_opt.dt = snap_dt(want);
    if (std::fabs(g_opt.dt - want) > 1e-4f) std::printf("  (%g ms divides 1 ms unevenly; using %g ms)\n", want, g_opt.dt);
    if (g_opt.dt > 0.1001f)
        std::printf("  Note: steps above 0.1 ms are faster but less accurate (the spike upstroke is very fast).\n");
    save_options();
    std::printf("  Saved: %d synapses per neuron, dt %g ms (%d steps per ms)\n",
                g_opt.syn, g_opt.dt, (int)std::lround(1.0 / g_opt.dt));
}

// ---------------------------------------------------------------------------
static void cmd_run() {
    const int N = ask("neurons", 1000);
    const float ms = ask("simulated time (ms)", 20000.0f);
    if (run_sim(cfg("run", N, ms)) == 0) { std::printf("\n"); plot_run(HERE); }
}

static void cmd_validate() {
    if (run_sim(cfg("validate", 1000, 2000.0f)) == 0) validate_output(HERE);
}

static void cmd_bench() {
    const int sizes[3] = { 10000, 30000, 100000 };
    const float ms = 5000.0f;
    std::vector<std::string> lines;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%8s %8s %8s %10s %10s %7s %8s   rates A/E/I (Hz)",
                  "neurons", "us/step", "sim (s)", "x realtime", "integrate", "sync", "deliver");
    lines.push_back(buf);
    for (int n : sizes) {
        RunResult r;
        if (run_sim(cfg("run", n, ms), &r) != 0 || !r.ok) {
            std::snprintf(buf, sizeof(buf), "%8d   (run failed)", n); lines.push_back(buf); continue;
        }
        std::snprintf(buf, sizeof(buf), "%8d %8.2f %8.2f %10.1f %10.2f %7.2f %8.2f   %.2f/%.2f/%.2f",
                      n, r.us_step, r.wall_s, 1000.0 * r.dt / r.us_step, r.integ_us, r.sync_us, r.deliv_us,
                      r.rate_A, r.rate_E, r.rate_I);
        lines.push_back(buf);
    }
    char opts[96];
    std::snprintf(opts, sizeof(opts), "dt %g ms, %d synapses per neuron", g_opt.dt, g_opt.syn);
    std::printf("\n================ BENCHMARK ================\n%s\n", opts);
    for (auto& l : lines) std::printf("%s\n", l.c_str());
    std::printf("===========================================\n");
    char date[64];
    const std::time_t now = std::time(nullptr);
    std::tm lt{};
#ifdef _WIN32
    localtime_s(&lt, &now);              // MSVC's safe version (plain localtime is an error with SDL checks)
#else
    localtime_r(&now, &lt);
#endif
    std::strftime(date, sizeof(date), "%Y-%m-%d %H:%M", &lt);
    std::ofstream log("bench_log.txt", std::ios::app);
    log << "\n=== " << date << "  Brain.exe v" << BRAIN_VERSION_STR << "  " << (int)(ms / 1000) << " s simulated per size  "
        << opts << " ===\n";
    for (auto& l : lines) log << l << "\n";
    std::printf("Appended to bench_log.txt\n");
}

static void cmd_seq() {
    RunConfig c = cfg("seq", ask("neurons", 10000), 1000.0f * ask("training time (s)", 60.0f));
    c.lr = ask("learning rate multiplier", 1.0f);
    if (run_sim(c) == 0) analyze_seq(HERE);
}

// Live views: continue a saved brain, or make a new one (and name it)
static void cmd_tone() {
    RunConfig c = cfg("tone", ask("neurons (at least 4,480)", 30000), 0.0f);
    c.tone_hz = ask("tone to train (Hz)", 500.0f);
    c.lr = ask("reward strength", 3.0f);
    c.tone_train = ask("training presentations (half are the trained tone)", 300);
    std::printf("  test 13 tones, train, test again: about %.0f s of simulated time\n",
                2.0 + (2 * 13 * 5 + c.tone_train) * 0.5);
    if (run_sim(c) == 0) analyze_tone(HERE);
}

// Walking creatures: several brains, each learning to walk a two-legged body
static void cmd_walk() {
    RunConfig c = cfg("walk", 0, 0.0f);
    c.brain_name = "walkers";
    std::vector<std::string> info;
    const std::vector<std::string> saved = list_brains(&info, 2);      // walker populations only
    if (!saved.empty()) {
        std::printf("\n  Saved walker populations:\n");
        for (size_t i = 0; i < saved.size(); ++i)
            std::printf("   %2zu) %-28s %s\n", i + 1, saved[i].c_str(), info[i].c_str());
        std::cout << "  Load which one? (number, or Enter for new creatures): " << std::flush;
        std::string line;
        std::getline(std::cin, line);
        try {
            const int k = line.empty() ? 0 : std::stoi(line);
            if (k >= 1 && k <= (int)saved.size()) {
                c.load_path = saved[k - 1];
                std::string stem = c.load_path;                // brains/name.brain -> name
                const size_t sl = stem.find_last_of("\\/");
                if (sl != std::string::npos) stem = stem.substr(sl + 1);
                if (stem.size() > 6 && stem.substr(stem.size() - 6) == ".brain") stem.resize(stem.size() - 6);
                c.brain_name = stem;
            }
        } catch (...) {}
    }
    if (c.load_path.empty()) {
        c.walk_brains = std::max(1, std::min(8, ask("creatures (brains), 1-8", 4)));
        c.walk_modular = ask("brain type: 1 = octopus (torso + a module per leg), 2 = one blob", 1) != 2;
        if (c.walk_modular) c.walk_cpg = ask("spinal rhythm generators (a walking rhythm per joint): 1 = yes, 2 = no", 1) != 2;
        const int lo = c.walk_modular ? 2000 : 1000;
        c.N = std::max(lo, (int)ask(c.walk_modular ? "neurons per brain (at least 2,000)" : "neurons per brain (at least 1,000)", 3000));
        std::cout << "  Name for saving these brains [walkers]: " << std::flush;
        std::string name;
        std::getline(std::cin, name);
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t\r") + 1);
        if (!name.empty()) c.brain_name = name;
        std::printf("  %d brains x %d neurons = %d neurons in total.\n", c.walk_brains, c.N, c.walk_brains * c.N);
    }
    std::printf("  (Save brains in the panel saves them as brains/%s.brain; closing does not save.\n"
                "   Close the window or press Esc to stop: the results are plotted and saved as walking.png.)\n",
                c.brain_name.c_str());
    if (run_sim(c) == 0) analyze_walk(HERE);
}

static void cmd_live(const char* extra, int default_n, const char* prompt) {
    RunConfig c = cfg("live", default_n, 0.0f);
    c.live_extra = extra;
    std::vector<std::string> info;
    const std::vector<std::string> brains = list_brains(&info, 1);    // live-view brains (not walkers)
    if (!brains.empty()) {
        std::printf("\n  Saved brains (only ones made in this same view will load):\n");
        for (size_t i = 0; i < brains.size(); ++i)
            std::printf("   %2zu) %-28s %s\n", i + 1, brains[i].c_str(), info[i].c_str());
        std::cout << "  Load which one? (number, or Enter for a new brain): " << std::flush;
        std::string line;
        std::getline(std::cin, line);
        try {
            const int k = line.empty() ? 0 : std::stoi(line);
            if (k >= 1 && k <= (int)brains.size()) {
                c.load_path = brains[k - 1];
                std::string stem = c.load_path;            // brains/name.brain -> name
                const size_t sl = stem.find_last_of("\\/");
                if (sl != std::string::npos) stem = stem.substr(sl + 1);
                if (stem.size() > 6 && stem.substr(stem.size() - 6) == ".brain") stem.resize(stem.size() - 6);
                c.brain_name = stem;
            }
        } catch (...) {}
    }
    if (c.load_path.empty()) {
        c.N = ask(prompt, default_n);
        std::cout << "  Name for the new brain [mybrain]: " << std::flush;
        std::string name;
        std::getline(std::cin, name);
        name.erase(0, name.find_first_not_of(" \t"));
        name.erase(name.find_last_not_of(" \t\r") + 1);
        if (!name.empty()) c.brain_name = name;
        std::printf("  (use Save brain in the panel to save it as brains/%s.brain; closing does not save)\n",
                    c.brain_name.c_str());
    }
    run_sim(c);
}

static void menu() {
    for (;;) {
        std::printf("\n=========== BRAIN SIMULATOR v%s ===========\n", BRAIN_VERSION_STR);
        std::printf("  1) Run simulation + plot\n");
        std::printf("  2) Validate against the CPU double-precision reference\n");
        std::printf("  3) Plot the latest run\n");
        std::printf("  4) Standard benchmark (10k / 30k / 100k neurons, 5 s each, logged)\n");
        std::printf("  5) Self-test the validator\n");
        std::printf("  6) Live view: brain in a vat\n");
        std::printf("  7) Sequence task: train A->B->C->D, test replay\n");
        std::printf("  8) Live view: sequence task\n");
        std::printf("  9) Live view: sound (ears on your speakers + a mouth)\n");
        std::printf(" 10) Live view: eyes (the brain watches your screen)\n");
        std::printf(" 11) Frequency task: reward one tone, measure the response before vs after\n");
        std::printf(" 12) Walking creatures: brains learn to walk, dopamine for going forward\n");
        std::printf(" 13) Brain options (now: %d synapses per neuron, dt %g ms)\n", g_opt.syn, g_opt.dt);
        std::printf("  q) Quit\n");
        std::printf("Choice: ");
        std::fflush(stdout);
        std::string c;
        if (!std::getline(std::cin, c)) return;
        c.erase(0, c.find_first_not_of(" \t"));
        c.erase(c.find_last_not_of(" \t\r") + 1);
        if (c == "1") cmd_run();
        else if (c == "2") cmd_validate();
        else if (c == "3") plot_latest(HERE);
        else if (c == "4") cmd_bench();
        else if (c == "5") validator_selftest();
        else if (c == "6") cmd_live("", 30000, "neurons");
        else if (c == "7") cmd_seq();
        else if (c == "8") cmd_live("seq", 30000, "neurons");
        else if (c == "9") cmd_live("ears+mouth", 30000, "neurons (at least 4,480)");
        else if (c == "10") cmd_live("eyes", 100000, "neurons (at least 25,920)");
        else if (c == "11") cmd_tone();
        else if (c == "12") cmd_walk();
        else if (c == "13") cmd_options();
        else if (c == "q" || c == "Q" || c == "quit" || c == "exit") return;
    }
}

int main(int argc, char** argv) {
    go_to_exe_folder();
    load_options();
    if (argc > 1) return run_sim(config_from_args(argc, argv));   // old command line, unchanged
    menu();
    return 0;
}
