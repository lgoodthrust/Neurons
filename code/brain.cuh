// Live visualizer switch: needs GLFW + GLEW (vcpkg). Comment out to build without it.
#include "version.h"          // BRAIN_VERSION: written to meta.txt; shown in bench_log.txt and plot titles

#define BRAIN_VIZ
// GLFW + GLEW linked statically into Brain.exe (vcpkg triplet x64-windows-static-md),
// so no DLLs are needed. Comment out if you link the DLL versions instead.
#define BRAIN_VIZ_STATIC
// On-screen control panel (sliders, buttons) in the live view. Needs Dear ImGui:
//   vcpkg install imgui[glfw-binding,opengl3-binding]:x64-windows-static-md
// Comment out to keep keyboard-only controls.
#define BRAIN_VIZ_UI

// =============================================================================
//  brain.cu  —  GPU spiking-network simulator (C++17 / CUDA)        v7: sparse, large-N, live view, sequence task
//
//  Model (same as v3/v4):
//    - N neurons: 80% excitatory (AdEx soma + 4 dendritic branches),
//      20% inhibitory (AdEx, soma only)
//    - ~80 local E and ~80 local I inputs per neuron (fixed in-degree), plus
//      external Poisson synapses up to ~6,900 total
//    - AMPA + NMDA (Mg block) + GABA_A conductance synapses, 1-3 ms delays
//    - STDP on E->E synapses
//    - Group A (first 25% of E cells) gets a clustered stimulus on branch 0
//
//  v5 design
//  ---------
//  * SPARSE synapses. Each neuron stores only its real incoming synapses, in
//    one compressed list (CSR) ordered by presynaptic id, so its E inputs come
//    first (for an E neuron those are exactly its plastic synapses):
//        in_ptr[n] .. in_eend[n]     E inputs
//        in_eend[n] .. in_ptr[n+1]   I inputs
//    Per synapse: presynaptic id, weight, packed (delay | branch << 6).
//    Memory ~2 KB per neuron, so 100,000+ neurons fit easily.
//  * MANY NEURONS PER THREAD. Neuron n belongs to thread n mod (total threads);
//    state is loaded at the start of every epoch and stored at the end, so the
//    neuron count is limited by memory, not by how many threads fit on the GPU.
//  * Min-delay epochs (as v4): integrate M = 10 steps with no communication,
//    one grid.sync(), then deliver.
//  * PULL delivery: each neuron scans its own ~160 inputs and checks whether the
//    presynaptic cell spiked this epoch (a 16-bit "which steps" word per
//    neuron). Cost per neuron is fixed, independent of N and of how many
//    neurons fire; every write goes to the neuron's own data, so no atomics.
//  * STDP exactly as v4: postsynaptic trace replayed from its epoch-start value,
//    presynaptic traces replayed from epoch-start value + spike bits.
//    Potentiation of a spiking neuron is shared by its warp (lanes take
//    consecutive CSR entries: coalesced).
//
//  v5.1: read-only synapse data goes through the read-only cache (__ldg).
//  v5.2: PUSH-assisted delivery. Scanning every input every epoch costs
//  N x 160 checks although only a few hundred neurons fire per epoch. Now each
//  epoch has two steps after integration:
//     push: every neuron that fired walks its OUTGOING synapse list and leaves a
//           note ("input q fired") with each target (atomic counter per target);
//     pull: each neuron processes only the notes it received, in the same way
//           as before (same delivery, same STDP order and arithmetic).
//  Work scales with spikes x 160 instead of N x 160. One extra grid barrier
//  per epoch. A neuron with more than MAXHIT notes falls back to the full scan.
//  v5.4: input prefetch. Nsight Compute showed ~42% of stalls were threads
//  waiting on memory. All 10 input slots an epoch reads were finalized in the
//  previous epoch, so step u+1's inputs are loaded while step u is computed.
//  v7.1: stateless random numbers. Philox is counter-based, so each neuron's
//  random numbers are computed from (neuron, step) instead of a 64-byte stored
//  state: no per-neuron RNG memory, no load/store each epoch, fewer registers.
//  Branch-voltage / plateau statistics are sampled once per epoch (1 ms) instead
//  of every step, which takes 8 accumulators out of the inner loop.
//  v10.1: menu 13 allows up to 10,000 synapses per neuron; above 2,600 the inhibitory ones stop at a real
//  neuron's 1,300 (background never negative); GPU memory is checked before the CPU builds the network.
//  v10.2: fix: runs longer than 59.6 simulated hours overflowed the 32-bit step counter (the brain froze,
//  then learning broke). Interactive views now rebase the counter every ~30 simulated hours.
//  v10.1: live views: 'Flat map' (M) unrolls the sphere like a world map (pan, zoom, right-click poke).
//  v10.0: spinal rhythm generators for octopus walkers: a half-centre oscillator per joint (flex/extend
//  groups exciting themselves, inhibiting each other through inhibitory pools, tired out by adaptation),
//  knee locked to hip, and a 'brainstem drive' slider. Designed and tuned on a CPU test bench first.
//  v9.9: real evolution for the walkers: each generation the weakest share is replaced by mutated
//  copies of the strongest (GPU mutation kernel, Philox noise); survivors unchanged; children start fresh.
//  v9.8: walkers save / load their whole population (menu 12) and plot + save walking.png
//  when the window closes.
//  v9.7: include only the Philox generator header (curand_kernel.h put ~1.3 MB of unused
//  generator tables into every .cu file: about 8 MB of Brain.exe).
//  v9.6: warp pool in pull: neurons that cannot be potentiated this epoch (no own E spike) have
//  all their notes processed by the whole warp together; their input-buffer adds become atomic.
//  Spiking E neurons keep the step-by-step path, so the STDP order is unchanged.
//  v9.5: pull prefetches the weights of each neuron's first two notes before the step loop
//  (NCU: 13% of GPU time was spent waiting for weights one note at a time). Same arithmetic.
//  v9.3: default back to 160 synapses per neuron (menu 13 sets any value).
//  v9.2: Brain options (menu 13): synapses per neuron and time step, saved in brain_options.txt;
//  saved brains remember both; the benchmark log records them.
//  v9.0: 300 simulated synapses per neuron everywhere (150 E + 150 I, was 80 + 80); the rest of
//  a neuron's ~6,900 inputs still arrive as background, so the total input is unchanged.
//  v8.6: octopus brains for the walkers: a torso module and one module per leg, wired mostly
//  inside themselves, with crossed inhibition between the legs. v8.5: human knees, stiff joints.
//  v8.3: walking creatures (menu 12): several brains, each driving a two-legged body
//  (Box2D) through sensory and motor neurons, learning from dopamine for moving forward.
//  v8.2: frequency task (menu 11). Tones go straight into the auditory nerve: test
//  13 tones, train one target tone with automatic dopamine after each presentation,
//  test again; A1 and mouth responses and pathway weights are compared (frequency.png).
//  v8.1: reward learning in the sound view. With "learn only from rewards" on, STDP
//  writes into a per-synapse eligibility trace (~1 s memory) instead of the weight;
//  pressing Reward (Enter) / Punish (Backspace) applies the recent traces as real
//  weight changes (three-factor, dopamine-modulated STDP, as in Izhikevich 2007).
//  v8.0: brains can be saved and loaded (.brain files in brains/ next to the exe):
//  wiring, learned weights, neuron state and layout, so learning carries over.
//  v7.9: sound-mode anatomy. In the live sound view the network is wired like a
//  (simplified) auditory system: auditory nerve -> A1 pitch maps on both sides ->
//  frontal mouth area, with distance-dependent local wiring everywhere. Other modes
//  keep the original random network.
//  v7.7: no divisions in the hot loops. Every x / tau * dt (and / C, / delta_T,
//  Mg / 3.57) is now x * (precomputed constant): a precise GPU division is a
//  multi-instruction routine, ~20 of them ran per neuron per step. Branch values
//  are loaded and stored as one float4 instead of four floats. Same equations;
//  only the last bit of rounding differs (validated against the CPU reference).
//  (v7.6, sparse input buffers, was measured slower and dropped.)
//  v7.5: everything in one program. Brain.exe with no arguments shows the menu that
//  brain.py used to: runs, plots (ImPlot windows, saved as PNG), the benchmark and log,
//  the validator (an independent CPU reference in double precision) and its self-test.
//  v7.4: ears and mouth. Ears hear what the speakers play (64 frequency bands x 32
//  neurons); the mouth is 48 groups of output neurons, each setting the loudness of
//  one tone of a synthesizer. Live mode "senses" = eyes + ears + mouth.
//  v7.3: eyes. In live mode with "eyes", the screen is captured, shrunk 10x
//  (1920x1080 -> 192x108) and each pixel's brightness drives one E neuron.
//  v7.2: delivery restructure. A note now carries everything the receiver needs:
//  the synapse index, the presynaptic neuron's spike bits and the synapse's delay /
//  branch / type (from a static copy in the outgoing list). Receiving a spike is
//  then one weight read plus the input-buffer update, instead of a chain of five
//  dependent reads (note -> presynaptic id -> its spike bits -> weight/meta).
//
//  v6: LIVE VISUALIZER ("brain in a vat"). Build with BRAIN_VIZ defined and
//  GLFW + GLEW available (see the setup notes in the reply / README), then run
//      Brain.exe 3.9 0 0.10 live 30000
//  Each frame the GPU advances the simulation, then a small kernel turns every
//  neuron's last spike time into a glow value written straight into an OpenGL
//  buffer (CUDA-OpenGL interop: the data never leaves the GPU). The simulation
//  code is unchanged; live mode only skips the file recordings.
//
//  v7: SEQUENCE TASK. Four groups A, B, C, D (each 10% of the E cells). Training
//  stimulates them in order, 15 ms apart, every 300 ms; STDP strengthens A->B,
//  B->C, C->D. Tests cue A alone (does B, C, D follow?) and D alone (control).
//      Brain.exe 3.9 <train_ms> 0.10 seq <N> [learning_rate]
//      Brain.exe 3.9 0 0.10 live <N> seq        (interactive: T trains, 1-4 cue)
//  Other modes are unchanged (the task is off, learning rate 1).
//
//  Output: meta.txt, params.txt, spikes.bin (int32 step, int32 neuron),
//    trace_soma.bin, trace_branch.bin, weights_mean.bin, Vs0.bin and, for
//    validate mode or N <= 20,000: csr_ptr.bin, csr_eend.bin, csr_pre.bin,
//    csr_meta.bin, csr_w_init.bin, csr_w_final.bin
//
//  Usage:  Brain.exe [ext_E_rate_Hz=3.9] [sim_ms=20000] [nmda_ratio=0.10]
//                    [run|validate] [N=1000]
// =============================================================================
//  Source layout:
//    brain.cuh   shared: switches, constants, Params / DevState, helpers, declarations
//    kernels.cu  the GPU simulation (neuron update, delivery, STDP, main kernel)
//    network.cu  building the network (random wiring, synapse lists)
//    viz.cu      live visualizer (window, camera, shaders, controls)
//    main.cu     program flow: arguments, parameters, running, statistics, output
// =============================================================================
#pragma once

#include <cuda_runtime.h>
#include <curand_philox4x32_x.h>   // Philox only: curand_kernel.h embeds ~220 KB of unused
                                   // generator tables into every file that includes it (v9.7)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>


#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",                \
                         cudaGetErrorString(err_), __FILE__, __LINE__);         \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ----------------------------- Constants ------------------------------------
constexpr int   B           = 4;      // dendritic branches per E cell
constexpr int   NT          = 3;      // recorded neurons: 0 (group A), N_A (other E), N_E (inhibitory)
constexpr int   L           = 64;     // input ring buffer length in steps (power of 2, >= M + max delay)
constexpr int   MAXHIT      = 24;     // spiking inputs remembered per neuron per epoch (more -> rescan)
constexpr float K_LOCAL_E   = 80.f;   // default local E inputs per neuron (menu 13 can change it)
constexpr float K_LOCAL_I   = 80.f;   // default local I inputs per neuron
constexpr int   CHUNK_STEPS = 10000;  // steps per kernel launch (keeps launches short for the Windows watchdog)
constexpr int   MAX_THREADS = 256;    // threads per block (max)
constexpr int   MIN_BLOCKS  = 3;      // blocks per SM the compiler must fit: 2 -> 128 regs, no spills, 33% occupancy;
                                      //                                      3 ->  85 regs, tiny spill, 50% occupancy

// ----------------------------- Parameters -----------------------------------
struct Params {
    int   N, N_E, N_A, M;
    float dt;
    float Cs_E, Cs_I, gLs_E, gLs_I, EL, VT, dT_E, dT_I, Vr_E, Vr_I;
    float a_E, b_E, tauw_E, tref_E, tref_I, V_peak;
    float Cb, gLb, g_cb, bAP, inh_soma_frac;
    float E_exc, E_inh, tau_ampa, tau_gaba, tau_nr, tau_nd, nmda_ratio, Mg;
    float g_exc, g_inh;
    float k_local;                             // simulated synapses per neuron (E + I), for saving
    unsigned rng_epoch;                        // bumped at each step-counter rebase so the noise never repeats (0 = original)
    float lamE_branch_E, lamE_branch_I, lamI, stim_lam;
    float expE_E, expE_I, expI, expStim;
    float tau_plus, tau_minus, A_plus, A_minus, w_max;
    // precomputed on the host (set_derived): the kernel multiplies instead of dividing
    float k_ampa, k_nr, k_nd, k_gaba, k_plus, k_minus;   // dt / tau
    float dtCs_E, dtCs_I, dtCb, dtTauw_E;                // dt / C, dt / tau_w
    float inv_dT_E, inv_dT_I, mg_c;                      // 1 / delta_T, Mg / 3.57
    int   stim_every, stim_len, warmup, rec_every, w_every;
    long long spike_cap;
    int   deterministic;
    int   record;          // 0 in live mode: no traces / weight snapshots (run length is open-ended)
    // live-view dials (neutral values in normal and validate runs)
    float inh_scale;       // multiplies all inhibitory synaptic input (1 = normal)
    int   auto_stim;       // 1 = group A gets its periodic stimulus
    // sequence task (seq = 0: off)
    int   seq, seq_K, seq_size;          // groups 0..K-1 = E neurons [g*size, (g+1)*size)
    int   seq_cycle, seq_delta, seq_len; // steps: training cycle, gap between groups, stimulus length
    int   train0, train1;                // training window [train0, train1) in steps
    int   seq_test, test0;
    // eyes (live mode): screen pixels drive E neurons 0 .. eye_w*eye_h-1, one pixel per neuron
    // reward learning (live sound view): STDP marks synapses as eligible instead of changing them;
    // a reward (dopamine) turns the recent marks into real weight changes
    int   da_mode;         // 1 = STDP goes into the eligibility traces, 0 = plain STDP
    float k_elig;          // dt / eligibility time constant
    int   eye_n;           // number of sensory neurons: eye pixels, then ear neurons (0 = senses off)
    float eye_gain;        // Poisson input per step on branch 0 at full strength (white pixel / loud band)               // seq_test = 1: outside training, cue A / D in alternate cycles (from test0)
};

struct DevState {
    // per-neuron state (loaded/stored every epoch)
    float *Vs, *w, *ref, *g_gaba, *x_pre, *y_post;
    float *Vb, *g_ampa, *n_r, *n_d;            // [N*B]
    float *vb_sum, *plateau;                   // [N*B] accumulated after warmup
    uint8_t* tflag;                            // [N] "spiked since last trace sample"
    // incoming synapses (CSR by postsynaptic neuron)
    int*      in_ptr;                          // [N+1]
    int*      in_eend;                         // [N]   end of E inputs of neuron n
    int*      in_pre;                          // [nnz] presynaptic id
    float*    in_w;                            // [nnz] weight (nS)
    uint16_t* in_meta;                         // [nnz] delay (bits 0-5) | branch << 6
    // outgoing synapses (CSR by presynaptic neuron), pointing into the incoming lists
    int*      out_ptr;                         // [N+1]
    int*      out_post;                        // [nnz] postsynaptic id
    int*      out_q;                           // [nnz] index of this synapse in the incoming CSR
    uint16_t* out_meta;                        // [nnz] delay | branch << 6 | (presynaptic is E) << 8
    // per-epoch notes: which incoming synapses of neuron n carried a spike
    int*      hitcnt;                          // [N]
    uint2*    hitn;                            // [N*MAXHIT] notes: x = synapse index q,
                                               //   y = presynaptic spike bits | out_meta << 16
    // delayed-input ring buffers
    float* buf_exc;                            // [L*N*B]
    float* buf_inh;                            // [L*N]
    // epoch exchange (double-buffered by epoch parity)
    uint16_t* ep_bits;                         // [2*N] which of the M steps each neuron spiked in
    int*      ep_cnt;                          // [2]   neurons that spiked this epoch
    int*      ep_id;                           // [2*N] their ids
    float*    x_start;                         // [2*N] presynaptic trace at epoch start
    float*    y_start;                         // [N]   postsynaptic trace at epoch start (own use)
    // recording
    unsigned long long* spike_count;           // [1]
    int*  spike_step;                          // [spike_cap]
    int*  spike_id;                            // [spike_cap]
    float *tr_s, *tr_b;                        // [n_rec*NT], [n_rec*NT*B]
    float *snapA, *snapO;                      // [n_snap*N]
    unsigned long long* prof;                  // [3] nanoseconds: integrate, grid sync wait, deliver
    int*  last_spike;                          // [N] step of each neuron's latest spike (for the visualizer)
    unsigned* spike_n;                         // [N] live view: spikes since the graphs last read them (else null)
    float* eye;                                // [eye_n] pixel brightness 0..1 (live eyes mode)
    float* elig;                               // [nnz] eligibility trace per synapse (reward learning, else null)
    int*   elig_t;                             // [nnz] step at which elig was last updated (it decays lazily)
    int*  poke_end;                            // [N] neuron gets stimulus input until this step (live poking)
    float* snapG;                              // [n_snap*K*K] sequence task: summed weights group q -> group p
};

// ----------------------------- Kernels (kernels.cu) -------------------------
__global__ void simulate(DevState s, Params p, int step_begin, int step_end);
// Reward (dopamine): apply each plastic synapse's eligibility (decayed to step `now`) times `amount`
__global__ void reward_kernel(const int* in_ptr, const int* in_eend, float* in_w, const float* elig,
                              const int* elig_t, int N_E, int now, float k_elig, float amount, float w_max);

// ----------------------------- Network (network.cu) -------------------------
struct Network {
    std::vector<int>      in_ptr, in_eend, in_pre;   // incoming CSR
    std::vector<float>    in_w;
    std::vector<uint16_t> in_meta;
    std::vector<int>      out_ptr, out_post, out_q;  // outgoing CSR
    std::vector<uint16_t> out_meta;                  // copy of in_meta per outgoing synapse + E flag
    std::vector<float>    Vs0, Vb0;                  // initial voltages
    long long cntA = 0, cntO = 0;                    // plastic synapses inside / outside group A
    size_t nnz = 0;
    std::vector<float>    xyzt;                      // anatomy only: neuron positions for the viewer
    std::vector<uint8_t>  region;                    // anatomy only: REG_* per neuron
    std::vector<int16_t>  band;                      // anatomy only: pitch band (nerve, A1) or tone (mouth), else -1
    // loaded brains only: the rest of the neuron state (a new brain starts these at zero)
    std::vector<float>    st_w, st_ref, st_gaba, st_x, st_y;      // [N]
    std::vector<float>    st_g_ampa, st_n_r, st_n_d;              // [N*B]
};
void finish_network(Network& net, int N, int N_E, int N_A);       // outgoing lists + counts

// ----------------------------- Brain files (brainfile.cu) -------------------
struct BrainInfo { int N = 0, N_E = 0, N_A = 0, senses = 0, anatomy = 0; size_t nnz = 0; double age_s = 0;
                   float dt = 0.1f, syn = 160.f;      // time step and synapse setting it was built with
                   int walk_K = 0, walk_n = 0, walk_mod = 0, walk_cpg = 0; };   // walker population (0 = not a walker file)
bool read_brain_header(const std::string& path, BrainInfo& info, std::string& err);
bool load_brain(const std::string& path, Network& net, BrainInfo& info, std::string& err);
// walk (optional): {creatures, neurons per brain, octopus 0/1, rhythm generators 0/1} for a walker file
bool write_brain_file(const std::string& path, const Network& net, int N, int N_E, int N_A, int senses,
                      double age_s, std::string& msg, float dt = 0.1f, float syn = 300.f,
                      const int* walk = nullptr);
bool save_brain(const std::string& path, const DevState& s, const Params& p, size_t nnz, int senses,
                const std::vector<float>* layout, double age_s, std::string& msg, const int* walk = nullptr);

// ----------------------------- Walking creatures (walker.cu) ---------------
// Several identical brains in one simulation (no connections between them), one per creature.
// Sensors: 24 groups of 20 excitatory cells; motors: 8 groups of 40 (flex / extend per joint).
//   blob brain:    one random network; sensors are local E cells [0, 480), motors the last 320
//   octopus brain: three modules - torso (balance), back leg, front leg - each wired mostly
//                  inside itself; each leg module senses and moves only its own leg
// Sense groups (pairs are + / -): 0-7 joint angles, 8-15 joint speeds (joints: back hip, back knee,
// front hip, front knee), 16-17 tilt, 18-19 spin, 20-21 forward speed, 22-23 back / front foot down.
constexpr int WALK_SENSE_GROUPS = 24, WALK_SENSE_PER = 20, WALK_MOTOR_GROUPS = 8, WALK_MOTOR_PER = 40;
constexpr int WALK_MIN_BLOB = 1000, WALK_MIN_OCTO = 2000;   // smallest brains (neurons)
struct WalkLayout {
    bool modular = false;
    int  sense_start[WALK_SENSE_GROUPS];   // first local E index of each sense group
    int  motor_start[WALK_MOTOR_GROUPS];   // first local E index of each motor group (2*joint + extend)
    int  modE[4], modI[4];                 // module boundaries (local E / local I): torso, back leg, front leg
};
WalkLayout walk_layout(int n, bool modular);
struct WalkCtx { int brains = 4; int n = 3000; size_t nnz = 0; bool modular = true; bool cpg = false;
                 double age_s = 0;              // simulated time the population had lived before (loaded)
                 std::string name = "walkers"; };   // saved as brains/<name>.brain
Network build_walk_network(const Params& p, int brains, int n, int d_min, int d_max, bool modular,
                           float kE, float kI,     // kE / kI: simulated E / I inputs per neuron
                           bool cpg = false);      // octopus only: spinal rhythm generators
int run_walk(DevState& s, Params& p, int blocks, int threads, const WalkCtx& ctx);

// Long runs: the step counter is a 32-bit int (it would overflow after 2^31 steps = 59.6 h at 0.1 ms).
// Interactive views call this before each launch: past 2^30 steps it subtracts a large shift from the
// counter and from every stored step (last spikes, poke ends, eligibility times). The shift is a multiple
// of the input ring, of two epochs and of the stimulus periods, so every schedule lines up afterwards.
void rebase_steps_if_needed(DevState& s, Params& p, int& step, size_t nnz);

// What the live view needs to save the brain it is running
struct LiveCtx {
    size_t nnz = 0;
    const std::vector<float>* layout = nullptr;   // anatomy positions (null = viewer's own layout)
    double age_s = 0;                             // simulated time this brain had lived before this session
    std::string name = "mybrain";                 // saved as brains/<name>.brain
};
struct Anatomy;
Network build_network(const Params& p, int N, int N_E, int N_A, int d_min, int d_max,
                      double p_from_E, double p_from_I, const Anatomy* anat = nullptr);

// Constants the kernel uses instead of divisions. Call after changing any parameter they come from.
inline void set_derived(Params& p) {
    p.k_ampa = p.dt / p.tau_ampa; p.k_nr = p.dt / p.tau_nr; p.k_nd = p.dt / p.tau_nd;
    p.k_gaba = p.dt / p.tau_gaba; p.k_plus = p.dt / p.tau_plus; p.k_minus = p.dt / p.tau_minus;
    p.dtCs_E = p.dt / p.Cs_E; p.dtCs_I = p.dt / p.Cs_I; p.dtCb = p.dt / p.Cb; p.dtTauw_E = p.dt / p.tauw_E;
    p.inv_dT_E = 1.0f / p.dT_E; p.inv_dT_I = 1.0f / p.dT_I; p.mg_c = p.Mg / 3.57f;
}

// Senses for the live view (bit mask): eyes = screen, ears = what the speakers play, mouth = synthesizer
enum { SENSE_EYES = 1, SENSE_EARS = 2, SENSE_MOUTH = 4 };
constexpr int EYE_W = 192, EYE_H = 108;          // screen / 10, one E neuron per pixel (neurons 0..)
constexpr int EAR_BANDS = 64, EAR_PER = 32;      // 64 frequency bands x 32 neurons (right after the eyes)
constexpr int MOUTH_TONES = 48, MOUTH_PER = 32;  // 48 tones x 32 neurons (the last E neurons)

// Sound-mode anatomy (live "ears+mouth"): neurons have positions on the sphere and wiring follows them.
//   auditory nerve: E neurons [0, ear_n), 64 bands x 32 fibres (half low-, half high-threshold);
//                   no excitatory input from the cortex, project by frequency to A1 on both sides
//   A1:             a patch on each side of the head, laid out as a pitch map (low pitch at the bottom)
//   mouth:          the last mouth_n E neurons, a frontal patch; part of their input comes from the
//                   A1 neurons tuned to the matching pitch (a simplified hearing -> speech pathway)
//   all cortex:     inputs mostly from nearby neurons (falling off with distance), some long-range;
//                   conduction delay grows with distance
struct Anatomy {
    bool on = false;
    int  ear_n = 0, mouth0 = 0, mouth_n = 0;
    float lambda = 0.20f;       // typical connection distance (radians on the unit sphere)
    float p_long = 0.15f;       // share of inputs picked from anywhere
    int   k_nerve = 24;         // nerve inputs per A1 neuron
    int   k_a1_mouth = 24;      // A1 inputs per mouth neuron
};
enum { REG_CORTEX = 0, REG_NERVE = 1, REG_A1 = 2, REG_MOUTH = 3 };

#ifdef BRAIN_VIZ
// ----------------------------- Live view (viz.cu) ---------------------------
// layout: optional neuron positions from the network builder (x, y, z, colour type) per neuron
int run_live(DevState& s, Params& p, int blocks, int threads, bool seq_live, int senses,
             const LiveCtx& ctx);
#endif

// =============================================================================
//  Host helpers
// =============================================================================
template <typename T>
T* dev_upload(const std::vector<T>& h) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)));
    if (!h.empty()) CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T>
T* dev_zeros(size_t count) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(count, 1) * sizeof(T)));
    CUDA_CHECK(cudaMemset(d, 0, std::max<size_t>(count, 1) * sizeof(T)));
    return d;
}
template <typename T>
std::vector<T> dev_download(const T* d, size_t count) {
    std::vector<T> h(count);
    if (count) CUDA_CHECK(cudaMemcpy(h.data(), d, count * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
template <typename T>
void write_bin(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}
