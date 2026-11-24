// sim.h - the simulator as a function the menu, benchmark and validator can call.
// Plain C++ (no CUDA types), so the .cpp files can include it.
#pragma once
#include <string>

struct RunConfig {
    float       rate = 3.9f;        // external E input rate (Hz)
    float       T_ms = 20000.0f;    // simulated time (ms); seq: training time
    float       nmda = 0.10f;       // NMDA/AMPA ratio
    std::string mode = "run";       // run | validate | live | seq
    int         N    = 1000;        // neurons
    float       lr   = 1.0f;        // seq: STDP learning-rate multiplier
    std::string live_extra;         // live: "seq", "eyes", "ears+mouth", "senses", ...
    std::string load_path;          // live: continue a saved brain (brains/<name>.brain) instead of building one
    std::string brain_name = "mybrain";   // live: name used when saving
    float       tone_hz = 500.0f;   // frequency task: the tone that is rewarded
    int         tone_train = 300;   // frequency task: training presentations (half are the target)
    int         walk_brains = 4;    // walking task: creatures (one brain each); N = neurons per brain
    bool        walk_modular = true;// walking task: octopus brain (torso + leg modules) instead of one blob
    // brain options (menu 13; the menu fills these in for every run)
    int         syn  = 160;         // simulated synapses per neuron (half E, half I); the rest of a
                                    // neuron's ~6,900 inputs arrive as background
    float       dt   = 0.1f;        // time step (ms); must divide 1 ms evenly (1/16 .. 1/2)
};

// Valid time steps divide 1 ms into a whole number of steps (2..16): 0.5, 0.25, 0.2, 0.125, 0.1, 0.0625 ...
inline float snap_dt(float dt) {
    if (!(dt > 0.f)) return 0.1f;
    int k = (int)(1.0f / dt + 0.5f);
    k = k < 2 ? 2 : k > 16 ? 16 : k;
    return 1.0f / (float)k;
}

struct RunResult {
    bool   ok = false;
    double wall_s = 0, us_step = 0;              // GPU wall time, microseconds per step
    double dt = 0.1;                             // the run's time step (ms)
    double integ_us = 0, sync_us = 0, deliv_us = 0;   // per-step kernel time split
    double rate_A = 0, rate_E = 0, rate_I = 0;   // firing rates after warm-up (Hz)
};

// Runs one simulation and writes the output files into the current folder.
// Returns 0 on success.
int run_sim(const RunConfig& c, RunResult* result = nullptr);

// Old command line: rate T_ms nmda mode N [extra]
RunConfig config_from_args(int argc, char** argv);

// Saved brains in brains/ (paths), with an optional one-line summary of each
#include <vector>
std::vector<std::string> list_brains(std::vector<std::string>* summaries = nullptr);
