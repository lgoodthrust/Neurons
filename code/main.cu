// main.cu - one simulation run: parameters, network, GPU run, statistics, output files.
// (The program's entry point, the menu, lives in app.cpp.)
#include "brain.cuh"
#include <cuda_profiler_api.h>   // cudaProfilerStart / Stop (profiling window)
#include <cstdlib>
#include "sim.h"

RunConfig config_from_args(int argc, char** argv) {
    RunConfig c;
    if (argc > 1) c.rate = std::stof(argv[1]);
    if (argc > 2) c.T_ms = std::stof(argv[2]);
    if (argc > 3) c.nmda = std::stof(argv[3]);
    if (argc > 4) c.mode = argv[4];
    if (argc > 5) c.N    = std::stoi(argv[5]);
    if (argc > 6) {
        if (c.mode == "seq" || c.mode == "tone") c.lr = std::stof(argv[6]);
        else c.live_extra = argv[6];
    }
    if (argc > 7 && c.mode == "tone") c.tone_hz = std::stof(argv[7]);
    return c;
}

// ---- frequency task: tone schedule ----
namespace {
struct ToneTrial { int start; int phase; int rewarded; float hz; int test_idx; };   // phase 0 test, 1 train, 2 test
const float TONE_MS = 200.f, GAP_MS = 300.f, TONE_WARM_MS = 2000.f;
const int   TONE_TESTS = 13, TONE_REPS = 5;           // test tones: target * 2^(k/2), k = -4 .. 8
float tone_test_hz(float target, int k) { return target * std::pow(2.f, (k - 4) * 0.5f); }
float tone_band(float hz) {                           // fractional ear band (0 .. 64) of a frequency
    return EAR_BANDS * std::log(hz / 20.f) / std::log(16000.f / 20.f) - 0.5f;
}
std::vector<ToneTrial> tone_schedule(float target, int n_train, float dt) {
    std::mt19937 g(5);
    std::vector<ToneTrial> t;
    auto test_block = [&](int phase) {
        std::vector<int> order;
        for (int r = 0; r < TONE_REPS; ++r) for (int k = 0; k < TONE_TESTS; ++k) order.push_back(k);
        std::shuffle(order.begin(), order.end(), g);
        for (int k : order) t.push_back({ 0, phase, 0, tone_test_hz(target, k), k });
    };
    test_block(0);
    for (int i = 0; i < n_train; ++i) {               // half target (rewarded), half other test tones
        if (i % 2 == 0) t.push_back({ 0, 1, 1, target, 4 });
        else { int k; do { k = (int)(g() % TONE_TESTS); } while (k == 4); t.push_back({ 0, 1, 0, tone_test_hz(target, k), k }); }
    }
    std::shuffle(t.begin() + TONE_TESTS * TONE_REPS, t.end(), g);
    test_block(2);
    const int warm = (int)std::lround(TONE_WARM_MS / dt), trial = (int)std::lround((TONE_MS + GAP_MS) / dt);
    for (size_t i = 0; i < t.size(); ++i) t[i].start = warm + (int)i * trial;
    return t;
}
}  // namespace

int run_sim(const RunConfig& cfg, RunResult* result) {
    if (result) *result = RunResult{};
    const float rate_ext_E = cfg.rate;                  // Hz
    float       T_ms       = cfg.T_ms;                  // ms (seq: training time)
    const float nmda_ratio = cfg.nmda;
    const bool  validate   = cfg.mode == "validate";
    const bool  live       = cfg.mode == "live";
    const bool  walk_run   = cfg.mode == "walk";          // walking creatures, one brain each
    const bool  interactive = live || walk_run;           // runs in a window until closed
    const int   walk_K     = std::max(1, cfg.walk_brains);
    const int   walk_n     = std::max(cfg.walk_modular ? WALK_MIN_OCTO : WALK_MIN_BLOB, cfg.N / 5 * 5);   // per brain (multiple of 5)
    const bool  seq_run    = cfg.mode == "seq";
    const bool  tone_run   = cfg.mode == "tone";          // frequency task
    const bool  seq_live   = live && cfg.live_extra == "seq";
    // live senses: "eyes", "ears", "mouth" (any combination, e.g. "eyes+ears") or "senses" = all three
    const std::string sense_arg = live ? cfg.live_extra : std::string();
    const int senses = tone_run ? (SENSE_EARS | SENSE_MOUTH)
                     : sense_arg == "senses" ? (SENSE_EYES | SENSE_EARS | SENSE_MOUTH)
                     : ((sense_arg.find("eyes") != std::string::npos ? SENSE_EYES : 0) |
                        (sense_arg.find("ears") != std::string::npos ? SENSE_EARS : 0) |
                        (sense_arg.find("mouth") != std::string::npos ? SENSE_MOUTH : 0));
    const bool  seq        = seq_run || seq_live;
    const float lr         = seq_run ? cfg.lr : 1.0f;   // STDP learning-rate multiplier
    const float pre_ms = 5000.f, post_ms = 5000.f, train_ms = seq_run ? T_ms : 0.f;
    if (seq_run) T_ms = pre_ms + train_ms + post_ms;     // test, train, test
    std::vector<ToneTrial> tones;
    if (tone_run) {
        tones = tone_schedule(cfg.tone_hz, std::max(2, cfg.tone_train), snap_dt(cfg.dt));
        T_ms = TONE_WARM_MS + tones.size() * (TONE_MS + GAP_MS);
    }
#ifndef BRAIN_VIZ
    if (interactive) {
        std::fprintf(stderr, "This Brain.exe was built without the visualizer (define BRAIN_VIZ and link GLFW + GLEW).\n");
        return 1;
    }
#endif
    // continuing a saved brain: its size and senses come from the file
    BrainInfo saved;
    if (!cfg.load_path.empty()) {
        std::string err;
        if (!live) { std::fprintf(stderr, "Saved brains can only be continued in a live view\n"); return 1; }
        if (!read_brain_header(cfg.load_path, saved, err)) { std::fprintf(stderr, "Cannot load brain: %s\n", err.c_str()); return 1; }
        if (saved.senses != senses) {
            std::fprintf(stderr, "That brain was made for a different view (senses %d, this view %d)\n", saved.senses, senses);
            return 1;
        }
    }
    const int   N          = walk_run ? walk_K * walk_n : cfg.load_path.empty() ? cfg.N : saved.N;
    if (N < 10) { std::fprintf(stderr, "N must be at least 10\n"); return 1; }
    const int N_E = (int)std::lround(0.8 * N);
    const int N_I = N - N_E;
    const int N_A = N_E / 4;
    const int n_eye   = (senses & SENSE_EYES)  ? EYE_W * EYE_H : 0;
    const int n_ear   = (senses & SENSE_EARS)  ? EAR_BANDS * EAR_PER : 0;
    const int n_mouth = (senses & SENSE_MOUTH) ? MOUTH_TONES * MOUTH_PER : 0;
    if (senses && N_E < n_eye + n_ear + n_mouth) {
        const int need = n_eye + n_ear + n_mouth;
        std::fprintf(stderr, "These senses need %d excitatory neurons: use N >= %d\n", need, (need * 5 + 3) / 4);
        return 1;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int coop = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0));
    const int SMs = prop.multiProcessorCount;
    std::printf("GPU: %s (compute %d.%d), %d SMs, %.1f GB, cooperative launch: %s\n",
                prop.name, prop.major, prop.minor, SMs, prop.totalGlobalMem / 1073741824.0,
                coop ? "yes" : "no");
    if (!coop) { std::fprintf(stderr, "This GPU/driver does not support cooperative launch.\n"); return 1; }
    if (validate) std::printf("VALIDATE MODE: deterministic inputs, every step recorded\n");
    if (seq_run)  std::printf("SEQUENCE TASK: %.0f s test, %.0f s training (A->B->C->D), %.0f s test; learning rate x%.2f\n",
                              pre_ms / 1000, train_ms / 1000, post_ms / 1000, lr);

    // ---------------- parameters ----------------
    Params p{};
    p.N = N; p.N_E = N_E; p.N_A = N_A;
    // time step and synapse setting: from the brain options, or from the saved brain being continued
    // (its delays are stored in steps, and its background input depends on its synapse count)
    p.dt = cfg.load_path.empty() ? snap_dt(cfg.dt) : saved.dt;
    const float syn = cfg.load_path.empty() ? (float)std::max(20, cfg.syn) : saved.syn;
    p.k_local = syn;
    if (!cfg.load_path.empty() && (std::fabs(saved.dt - snap_dt(cfg.dt)) > 1e-6f || std::fabs(saved.syn - cfg.syn) > 0.5f))
        std::printf("Note: this brain was built with dt %g ms and %.0f synapses per neuron; it keeps those "
                    "(brain options: dt %g ms, %d synapses)\n", saved.dt, saved.syn, snap_dt(cfg.dt), cfg.syn);
    p.Cs_E = 100.f; p.Cs_I = 200.f; p.gLs_E = 5.f; p.gLs_I = 20.f;
    p.EL = -70.f; p.VT = -50.f; p.dT_E = 2.f; p.dT_I = 0.5f;
    p.Vr_E = -58.f; p.Vr_I = -60.f;
    p.a_E = 2.f; p.b_E = 60.f; p.tauw_E = 200.f; p.tref_E = 2.f; p.tref_I = 1.f;
    p.V_peak = 0.f;
    p.Cb = 100.f; p.gLb = 10.f; p.g_cb = 30.f; p.bAP = 8.f; p.inh_soma_frac = 0.3f;
    p.E_exc = 0.f; p.E_inh = -80.f;
    p.tau_ampa = 5.f; p.tau_gaba = 10.f; p.tau_nr = 2.f; p.tau_nd = 100.f;
    p.nmda_ratio = nmda_ratio; p.Mg = 1.f;
    p.g_exc = 0.35f; p.g_inh = 0.8f;

    // simulated (local) inputs per neuron; the rest of a cortical neuron's ~6,900 synapses
    // (5,600 E + 1,300 I) arrive as background input. Walkers can have more simulated ones.
    const float kLocE = 0.5f * syn, kLocI = 0.5f * syn;
    const double p_from_E = std::min(1.0, (double)kLocE / N_E);
    const double p_from_I = std::min(1.0, (double)kLocI / N_I);
    const float K_ext_E = 5600.f - kLocE, K_ext_I = 1300.f - kLocI;
    const float rate_ext_I = 2.4f, ext_E_scale_I = 0.55f;
    p.lamE_branch_E = K_ext_E * rate_ext_E * p.dt / 1000.f / B;
    p.lamE_branch_I = p.lamE_branch_E * ext_E_scale_I;
    p.lamI          = K_ext_I * rate_ext_I * p.dt / 1000.f;
    p.stim_lam      = 40.f * p.dt;
    p.expE_E  = std::exp(-p.lamE_branch_E);
    p.expE_I  = std::exp(-p.lamE_branch_I);
    p.expI    = std::exp(-p.lamI);
    p.expStim = std::exp(-p.stim_lam);

    p.tau_plus = 17.f; p.tau_minus = 34.f;
    set_derived(p);                                       // dt/tau etc. used by the kernel
    p.A_plus = 0.010f * p.g_exc * lr; p.A_minus = 0.0053f * p.g_exc * lr; p.w_max = 8.f * p.g_exc;

    const int d_min = (int)std::lround(1.0f / p.dt), d_max = (int)std::lround(3.0f / p.dt);
    p.M = d_min;
    if (p.M > 16 || p.M + d_max > L || d_max > 63) { std::fprintf(stderr, "delay settings incompatible with buffers\n"); return 1; }
    const int steps = ((int)std::lround(T_ms / p.dt) + p.M - 1) / p.M * p.M;
    p.stim_every = (int)std::lround(200.f / p.dt);
    p.stim_len   = (int)std::lround(10.f / p.dt);
    p.warmup     = (int)std::lround(500.f / p.dt);
    p.rec_every  = validate ? 1 : (int)std::lround(1.f / p.dt);
    p.w_every    = (int)std::lround(250.f / p.dt);
    p.deterministic = validate ? 1 : 0;
    p.record        = interactive ? 0 : 1;
    p.inh_scale     = 1.0f;
    p.auto_stim     = (tone_run || walk_run) ? 0 : 1;      // tasks: their own inputs are the only stimulus
    // sequence task
    p.seq       = seq ? 1 : 0;
    p.seq_K     = 4;
    p.seq_size  = N_E / 10;
    p.seq_cycle = (int)std::lround(300.f / p.dt);
    p.seq_delta = (int)std::lround(15.f / p.dt);
    p.seq_len   = (int)std::lround(8.f / p.dt);
    p.seq_test  = seq_run ? 1 : 0;
    p.test0     = (int)std::lround(1000.f / p.dt);               // skip the start-up transient
    p.train0    = seq_run ? (int)std::lround(pre_ms / p.dt) : 0;
    p.train1    = seq_run ? (int)std::lround((pre_ms + train_ms) / p.dt) : 0;   // live: T key opens it
    const double T_s = steps * p.dt / 1000.0;
    p.spike_cap  = interactive ? 0 : (long long)std::min(3.0e8, N * T_s * 25.0 + 200000.0);   // room for 25 Hz average
    const int n_rec      = std::max(1, (steps + p.rec_every - 1) / p.rec_every);
    const int n_snap_run = std::max(1, (steps + p.w_every - 1) / p.w_every);
    const int n_snap     = n_snap_run + 1;

    // ---------------- build the network (incoming + outgoing CSR) ----------------
    // sound mode (live ears, no eyes): wire the network like a simplified auditory system
    Anatomy anat;
    anat.on      = (live || tone_run) && (senses & SENSE_EARS) && !(senses & SENSE_EYES);
    anat.ear_n   = n_ear;
    anat.mouth_n = n_mouth;
    anat.mouth0  = N_E - n_mouth;
    Network net;
    if (!cfg.load_path.empty()) {
        std::string err;
        if (!load_brain(cfg.load_path, net, saved, err)) { std::fprintf(stderr, "Cannot load brain: %s\n", err.c_str()); return 1; }
        if (saved.N_E != N_E || saved.N_A != N_A) { std::fprintf(stderr, "Brain file does not match this build\n"); return 1; }
    } else if (walk_run) {
        net = build_walk_network(p, walk_K, walk_n, d_min, d_max, cfg.walk_modular, kLocE, kLocI);
    } else {
        net = build_network(p, N, N_E, N_A, d_min, d_max, p_from_E, p_from_I, &anat);
    }
    auto& in_ptr = net.in_ptr; auto& in_eend = net.in_eend; auto& in_pre = net.in_pre;
    auto& in_w = net.in_w; auto& in_meta = net.in_meta;
    auto& out_ptr = net.out_ptr; auto& out_post = net.out_post; auto& out_q = net.out_q;
    auto& Vs0 = net.Vs0; auto& Vb0 = net.Vb0;
    const long long cntA = net.cntA, cntO = net.cntO;
    const size_t nnz = net.nnz;

    // ---------------- memory check ----------------
    const double mem =
        nnz * 20.0 + (N + 1) * 12.0 +                                 // incoming + outgoing CSR
        (double)N * (MAXHIT * 8.0 + 4.0) + 2.0 * N * 4.0 +           // notes + epoch spike list
        (double)L * N * (B + 1) * 4.0 +                               // ring buffers
        N * (6 * 4.0 + 6 * B * 4.0 + 1 + 2 * 2 + 3 * 4) +        // state + epoch exchange
        p.spike_cap * 8.0 + (double)n_snap * N * 8.0 + (double)n_rec * NT * (B + 1) * 4.0;
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("  GPU memory needed: %.2f GB (free: %.2f GB)\n", mem / 1073741824.0, free_b / 1073741824.0);
    if (mem > 0.95 * free_b) { std::fprintf(stderr, "Not enough GPU memory for this N / duration.\n"); return 1; }

    // ---------------- launch geometry ----------------
    int per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, simulate, MAX_THREADS, 0));
    if (per_sm < 1) { std::fprintf(stderr, "Kernel cannot be resident (occupancy 0).\n"); return 1; }
    // Spread neurons evenly. First the number of rounds (neurons per thread) the GPU's resident
    // threads require; then the smallest block size, in warps of 32, that covers N in exactly that
    // many rounds while staying within k blocks per SM (smallest k first). Every round is then
    // (nearly) full, so no warp idles through a mostly-empty last round.
    const int max_resident = per_sm * SMs * MAX_THREADS;
    const int rounds_needed = (N + max_resident - 1) / max_resident;
    int threads = MAX_THREADS, blocks = per_sm * SMs;
    for (int k = 1; k <= per_sm; ++k) {
        const int t = ((N + rounds_needed * k * SMs - 1) / (rounds_needed * k * SMs) + 31) / 32 * 32;
        if (t <= MAX_THREADS) {
            threads = std::max(t, 32);
            blocks  = (N + rounds_needed * threads - 1) / (rounds_needed * threads);
            break;
        }
    }
    int per_sm_t = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm_t, simulate, threads, 0));
    if (blocks > per_sm_t * SMs) blocks = per_sm_t * SMs;
    const int rounds = (N + blocks * threads - 1) / (blocks * threads);
    std::printf("Launch: %d blocks x %d threads, %d neuron(s) per thread\n", blocks, threads, rounds);

    // ---------------- upload ----------------
    DevState s{};
    s.Vs = dev_upload(Vs0);        s.Vb = dev_upload(Vb0);
    // a new brain starts these at zero; a loaded brain continues from its saved state
    auto state = [&](const std::vector<float>& v, size_t n) { return v.size() == n ? dev_upload(v) : dev_zeros<float>(n); };
    s.w = state(net.st_w, N);       s.ref = state(net.st_ref, N);   s.g_gaba = state(net.st_gaba, N);
    s.x_pre = state(net.st_x, N);   s.y_post = state(net.st_y, N);
    s.g_ampa = state(net.st_g_ampa, (size_t)N * B); s.n_r = state(net.st_n_r, (size_t)N * B);
    s.n_d = state(net.st_n_d, (size_t)N * B);
    s.vb_sum = dev_zeros<float>((size_t)N * B); s.plateau = dev_zeros<float>((size_t)N * B);
    s.tflag = dev_zeros<uint8_t>(N);
    s.in_ptr = dev_upload(in_ptr); s.in_eend = dev_upload(in_eend); s.in_pre = dev_upload(in_pre);
    s.in_w = dev_upload(in_w);     s.in_meta = dev_upload(in_meta);
    s.out_ptr = dev_upload(out_ptr); s.out_post = dev_upload(out_post); s.out_q = dev_upload(out_q);
    s.out_meta = dev_upload(net.out_meta);
    s.hitcnt = dev_zeros<int>(N);    s.hitn = dev_zeros<uint2>((size_t)N * MAXHIT);
    s.ep_cnt = dev_zeros<int>(2);    s.ep_id = dev_zeros<int>((size_t)2 * N);
    s.buf_exc = dev_zeros<float>((size_t)L * N * B);
    s.buf_inh = dev_zeros<float>((size_t)L * N);
    s.ep_bits = dev_zeros<uint16_t>((size_t)2 * N);
    s.x_start = dev_zeros<float>((size_t)2 * N);
    s.y_start = dev_zeros<float>(N);
    s.spike_count = dev_zeros<unsigned long long>(1);
    s.spike_step  = dev_zeros<int>((size_t)p.spike_cap);
    s.spike_id    = dev_zeros<int>((size_t)p.spike_cap);
    s.tr_s    = dev_zeros<float>((size_t)n_rec * NT);
    s.tr_b    = dev_zeros<float>((size_t)n_rec * NT * B);
    s.snapA   = dev_zeros<float>((size_t)n_snap * N);
    s.snapO   = dev_zeros<float>((size_t)n_snap * N);
    s.snapG   = dev_zeros<float>((size_t)n_snap * 16);
    s.prof    = dev_zeros<unsigned long long>(3);
    s.last_spike = dev_upload(std::vector<int>(N, -1000000000));
    s.spike_n    = interactive ? dev_zeros<unsigned>(N) : nullptr;
    // reward learning: sound view only (eligibility trace + timestamp per synapse, 8 bytes each)
    const bool reward_ok = ((live || tone_run) && (senses & SENSE_EARS)) || walk_run;
    s.elig   = reward_ok ? dev_zeros<float>(nnz) : nullptr;
    s.elig_t = reward_ok ? dev_zeros<int>(nnz) : nullptr;
    p.da_mode = reward_ok ? 1 : 0;                       // sound view: learn only from rewards
    p.k_elig  = p.dt / 1000.0f;                          // 1 s memory
    s.poke_end   = dev_upload(std::vector<int>(N, -1));
    p.eye_n = walk_run ? N_E : n_eye + n_ear;            // sensory input: pixels + ear bands (walk: any E cell)
    p.eye_gain = p.stim_lam;                             // full strength = same drive as a group-A stimulus
    s.eye = p.eye_n ? dev_zeros<float>(p.eye_n) : nullptr;
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef BRAIN_VIZ
    if (walk_run) {
        WalkCtx wc; wc.brains = walk_K; wc.n = walk_n; wc.nnz = nnz; wc.modular = cfg.walk_modular;
        const int rc = run_walk(s, p, blocks, threads, wc);
        cudaDeviceReset();
        return rc;
    }
    if (live) {
        const int rc = run_live(s, p, blocks, threads, seq_live, senses, [&]() {
            LiveCtx c;
            c.nnz = nnz; c.layout = net.xyzt.empty() ? nullptr : &net.xyzt;
            c.age_s = cfg.load_path.empty() ? 0.0 : saved.age_s; c.name = cfg.brain_name;
            return c;
        }());
        cudaDeviceReset();
        return rc;
    }
#endif

    // ---------------- run ----------------
    std::printf("Simulating %d neurons for %.1f s (%d steps, dt = %.2f ms, epochs of %d steps)...\n",
                N, T_s, steps, p.dt, p.M);
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto run_steps = [&](int from, int to) {
        const int chunk = std::max(1, CHUNK_STEPS / p.M) * p.M;    // whole epochs per launch
        for (int begin = from; begin < to; begin += chunk) {
            int end = std::min(begin + chunk, to);
            void* args[] = { &s, &p, &begin, &end };
            CUDA_CHECK(cudaLaunchCooperativeKernel((void*)simulate, dim3(blocks), dim3(threads), args, 0, 0));
            CUDA_CHECK(cudaGetLastError());
        }
    };
    // Profiling window for Nsight Compute (ncu --profile-from-start off): the 10 epochs right after
    // the warm-up run as one short launch between cudaProfilerStart / Stop, so only they are
    // profiled. Without a profiler these calls do nothing, and the extra launch boundary does not
    // change the results. BRAIN_PROFILE=<epochs> changes the length; BRAIN_PROFILE=0 turns it off.
    const char* prof_env = std::getenv("BRAIN_PROFILE");
    int prof_epochs = prof_env ? std::max(0, std::atoi(prof_env)) : 10;
    if (!tone_run && prof_epochs > 0 && steps > p.warmup + p.M) {
        const int a = (p.warmup + p.M - 1) / p.M * p.M;            // first epoch after the warm-up
        const int b = std::min(steps, a + prof_epochs * p.M);
        run_steps(0, a);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("Profiling window: steps %d-%d (%d epochs, %.1f ms of simulated time)\n",
                    a, b, (b - a) / p.M, (b - a) * p.dt);
        cudaProfilerStart();
        run_steps(a, b);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaProfilerStop();
        run_steps(b, steps);
    } else if (!tone_run) {
        run_steps(0, steps);
    } else {
        // frequency task: each trial = tone (straight into the auditory nerve), then silence;
        // after every target tone in training, a dopamine pulse applies the recent eligibility
        const int tone_steps = (int)std::lround(TONE_MS / p.dt), trial_steps = (int)std::lround((TONE_MS + GAP_MS) / p.dt);
        std::vector<float> silent((size_t)p.eye_n, 0.f), tone_in((size_t)p.eye_n, 0.f);
        CUDA_CHECK(cudaMemcpy(s.eye, silent.data(), silent.size() * 4, cudaMemcpyHostToDevice));
        run_steps(0, tones.front().start);
        int done = 0;
        for (const ToneTrial& tr : tones) {
            const float c = tone_band(tr.hz);         // a pure tone excites its band and, less, its neighbours
            for (int i = 0; i < p.eye_n; ++i) {
                const float d = (i / EAR_PER) - c;
                tone_in[i] = std::exp(-d * d / (2.f * 0.8f * 0.8f));
            }
            CUDA_CHECK(cudaMemcpy(s.eye, tone_in.data(), tone_in.size() * 4, cudaMemcpyHostToDevice));
            run_steps(tr.start, tr.start + tone_steps);
            if (tr.rewarded)
                reward_kernel<<<(N_E + 127) / 128, 128>>>(s.in_ptr, s.in_eend, s.in_w, s.elig, s.elig_t, N_E,
                                                         tr.start + tone_steps, p.k_elig, cfg.lr, p.w_max);
            CUDA_CHECK(cudaMemcpy(s.eye, silent.data(), silent.size() * 4, cudaMemcpyHostToDevice));
            run_steps(tr.start + tone_steps, tr.start + trial_steps);
            if (++done % 50 == 0) { std::printf("\r  trial %d / %zu", done, tones.size()); std::fflush(stdout); }
        }
        std::printf("\r  %zu trials done            \n", tones.size());
        run_steps(tones.back().start + trial_steps, steps);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double wall = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

    // ---------------- download ----------------
    unsigned long long n_spk_all = 0;
    CUDA_CHECK(cudaMemcpy(&n_spk_all, s.spike_count, sizeof(n_spk_all), cudaMemcpyDeviceToHost));
    size_t n_spk = (size_t)std::min<unsigned long long>(n_spk_all, (unsigned long long)p.spike_cap);
    if (n_spk < n_spk_all)
        std::printf("WARNING: %llu spikes, only the first %zu were recorded\n", n_spk_all, n_spk);
    auto spike_step = dev_download(s.spike_step, n_spk);
    auto spike_id   = dev_download(s.spike_id, n_spk);
    auto tr_s    = dev_download(s.tr_s, (size_t)n_rec * NT);
    auto tr_b    = dev_download(s.tr_b, (size_t)n_rec * NT * B);
    auto prof    = dev_download(s.prof, 3);
    auto snapA   = dev_download(s.snapA, (size_t)n_snap * N);
    auto snapO   = dev_download(s.snapO, (size_t)n_snap * N);
    auto snapG   = dev_download(s.snapG, (size_t)n_snap * 16);
    auto vb_sum  = dev_download(s.vb_sum, (size_t)N * B);
    auto plateau = dev_download(s.plateau, (size_t)N * B);
    auto w_final = dev_download(s.in_w, nnz);

    for (int i = 0; i < N_E; ++i) {                    // final snapshot from final weights
        float sA = 0, sO = 0;
        for (int q = in_ptr[i]; q < in_eend[i]; ++q) {
            if (i < N_A && in_pre[q] < N_A) sA += w_final[q]; else sO += w_final[q];
        }
        snapA[(size_t)n_snap_run * N + i] = sA;
        snapO[(size_t)n_snap_run * N + i] = sO;
    }
    // sequence task: mean weight group q -> group p over time (last snapshot = final weights)
    std::vector<float> group_w;
    if (seq) {
        const int K = p.seq_K, S = p.seq_size;
        std::vector<double> cnt(K * K, 0.0), fin(K * K, 0.0);
        for (int i = 0; i < K * S; ++i)
            for (int q = in_ptr[i]; q < in_eend[i]; ++q)
                if (in_pre[q] < K * S) {
                    const int k = (in_pre[q] / S) * K + i / S;
                    cnt[k] += 1; fin[k] += w_final[q];
                }
        group_w.resize((size_t)n_snap * K * K);
        for (int sn = 0; sn < n_snap; ++sn)
            for (int k = 0; k < K * K; ++k) {
                const double sum = (sn == n_snap - 1) ? fin[k] : snapG[(size_t)sn * 16 + k];
                group_w[(size_t)sn * K * K + k] = (float)(cnt[k] > 0 ? sum / cnt[k] : 0.0);
            }
        const auto gw = [&](int sn, int a, int b) { return group_w[(size_t)sn * K * K + a * K + b]; };
        std::printf("Sequence task: mean weight (nS)       start -> end\n");
        for (int a = 0; a + 1 < K; ++a)
            std::printf("  %c -> %c (forward)   %.3f -> %.3f     %c -> %c (reverse)  %.3f -> %.3f\n",
                        'A' + a, 'A' + a + 1, gw(0, a, a + 1), gw(n_snap - 1, a, a + 1),
                        'A' + a + 1, 'A' + a, gw(0, a + 1, a), gw(n_snap - 1, a + 1, a));
    }

    std::vector<float> w_mean(2 * (size_t)n_snap);
    for (int k = 0; k < n_snap; ++k) {
        double a = 0, o = 0;
        for (int i = 0; i < N; ++i) { a += snapA[(size_t)k * N + i]; o += snapO[(size_t)k * N + i]; }
        w_mean[2 * k]     = (float)(cntA ? a / cntA : 0.0);
        w_mean[2 * k + 1] = (float)(cntO ? o / cntO : 0.0);
    }

    // ---------------- statistics ----------------
    const double T_sim = steps * p.dt;
    const double dur = (T_sim - 500.0) / 1000.0;
    double cA = 0, cE = 0, cI = 0;
    for (size_t q = 0; q < n_spk; ++q) {
        if (spike_step[q] < p.warmup) continue;
        const int n = spike_id[q];
        (n < N_A ? cA : n < N_E ? cE : cI) += 1;
    }
    const double avg_steps = (double)(steps - p.warmup) / p.M;   // branch statistics: one sample per epoch
    double vb_mean[B] = {}, plat_pct[B] = {};
    for (int k = 0; k < B; ++k) {
        for (int n = 0; n < N_E; ++n) vb_mean[k] += vb_sum[(size_t)n * B + k] / avg_steps / N_E;
        for (int n = 0; n < N_A; ++n) plat_pct[k] += 100.0 * plateau[(size_t)n * B + k] / avg_steps / N_A;
    }
    const double wA0 = w_mean[0], wA1 = w_mean[2 * (n_snap - 1)];
    const double wO0 = w_mean[1], wO1 = w_mean[2 * (n_snap - 1) + 1];

    if (result) {
        result->ok = true; result->wall_s = wall; result->us_step = 1e6 * wall / steps; result->dt = p.dt;
        result->integ_us = prof[0] / 1000.0 / steps; result->sync_us = prof[1] / 1000.0 / steps;
        result->deliv_us = prof[2] / 1000.0 / steps;
        result->rate_A = cA / N_A / dur; result->rate_E = cE / (N_E - N_A) / dur; result->rate_I = cI / N_I / dur;
    }
    std::printf("Spikes recorded: %zu\n", n_spk);
    std::printf("Firing rate  group A: %.2f Hz   other E: %.2f Hz   I: %.2f Hz\n",
                cA / N_A / dur, cE / (N_E - N_A) / dur, cI / N_I / dur);
    std::printf("Mean branch voltage (all E cells):");
    for (int k = 0; k < B; ++k) std::printf("  b%d %.1f", k, vb_mean[k]);
    std::printf(" mV\nGroup A time in NMDA plateau (> -35 mV):");
    for (int k = 0; k < B; ++k) std::printf("  b%d %.1f%%", k, plat_pct[k]);
    std::printf("\nE->E weights inside group A: %.3f -> %.3f nS (%+.0f%%)   [%lld synapses]\n",
                wA0, wA1, 100 * (wA1 / wA0 - 1), cntA);
    std::printf("E->E weights elsewhere:      %.3f -> %.3f nS (%+.0f%%)   [%lld synapses]\n",
                wO0, wO1, 100 * (wO1 / wO0 - 1), cntO);
    std::printf("sim time: %.3f s  (%.4f s per simulated second, %.2f us per step)\n",
                wall, wall / T_s, 1e6 * wall / steps);
    {
        const double tot = (double)(prof[0] + prof[1] + prof[2]);
        std::printf("Kernel time split (thread 0): integrate %.0f%%, grid sync wait %.0f%%, deliver/STDP %.0f%%\n",
                    100.0 * prof[0] / tot, 100.0 * prof[1] / tot, 100.0 * prof[2] / tot);
        std::printf("  per step: integrate %.2f us, grid sync wait %.2f us, deliver/STDP %.2f us\n",
                    prof[0] / 1000.0 / steps, prof[1] / 1000.0 / steps, prof[2] / 1000.0 / steps);
        std::printf("Avg spikes per step: %.2f\n", (double)n_spk_all / steps);
    }

    // ---------------- write output ----------------
    const bool full = validate || N <= 20000;
    {
        std::ofstream m("meta.txt");
        m << "version " BRAIN_VERSION "\nmode " << (validate ? 1 : 0) << "\nn_trace " << NT
          << "\ntrace_id0 0\ntrace_id1 " << N_A << "\ntrace_id2 " << N_E << "\nN " << N
          << "\nN_E " << N_E << "\nN_A " << N_A << "\nB " << B
          << "\nsteps " << steps << "\ndt " << p.dt << "\nT_ms " << T_sim
          << "\nrec_every " << p.rec_every << "\nw_every " << p.w_every
          << "\nn_rec " << n_rec << "\nn_snap " << n_snap << "\nn_spikes " << n_spk
          << "\nnnz " << nnz << "\nsyn_setting " << p.k_local << "\nstim_every " << p.stim_every << "\nstim_len " << p.stim_len
          << "\nwarmup " << p.warmup << "\nblocks " << blocks << "\nthreads " << threads
          << "\nfull_weights " << (full ? 1 : 0) << "\nwall_s " << wall
          << "\nseq " << p.seq << "\nseq_K " << p.seq_K << "\nseq_size " << p.seq_size
          << "\nseq_cycle " << p.seq_cycle << "\nseq_delta " << p.seq_delta << "\nseq_len " << p.seq_len
          << "\ntrain0 " << p.train0 << "\ntrain1 " << p.train1 << "\ntest0 " << p.test0
          << "\nlr " << lr << "\n";
        if (tone_run)
            m << "tone 1\ntone_hz " << cfg.tone_hz << "\ntone_steps " << (int)std::lround(TONE_MS / p.dt)
              << "\ngap_steps " << (int)std::lround(GAP_MS / p.dt) << "\ntone_tests " << TONE_TESTS
              << "\ntone_reps " << TONE_REPS << "\ntone_train " << cfg.tone_train << "\nreward " << cfg.lr
              << "\near_n " << n_ear << "\nmouth0 " << (N_E - n_mouth) << "\nmouth_n " << n_mouth << "\n";
    }
    {
        std::vector<int32_t> sp(2 * n_spk);
        for (size_t q = 0; q < n_spk; ++q) { sp[2 * q] = spike_step[q]; sp[2 * q + 1] = spike_id[q]; }
        write_bin("spikes.bin", sp);
    }
    write_bin("trace_soma.bin", tr_s);
    write_bin("trace_branch.bin", tr_b);
    write_bin("weights_mean.bin", w_mean);
    if (seq) write_bin("group_weights.bin", group_w);
    if (tone_run) {
        std::vector<float> tt;                        // start, phase, rewarded, Hz, test index
        for (const ToneTrial& tr : tones) { tt.push_back((float)tr.start); tt.push_back((float)tr.phase);
            tt.push_back((float)tr.rewarded); tt.push_back(tr.hz); tt.push_back((float)tr.test_idx); }
        write_bin("tone_trials.bin", tt);
        write_bin("region.bin", net.region);
        write_bin("band.bin", net.band);
        // mean weight of each pathway by pitch, before and after: nerve -> A1 (by A1 band), A1 -> mouth (by tone)
        std::vector<double> sum(4 * 64, 0.0), cnt(4 * 64, 0.0);
        for (int i = 0; i < N_E; ++i) {
            const int ri = net.region[i];
            if (ri != REG_A1 && ri != REG_MOUTH) continue;
            for (int q = in_ptr[i]; q < in_eend[i]; ++q) {
                const int j = in_pre[q];
                if (ri == REG_A1 && net.region[j] == REG_NERVE) {
                    const int b = net.band[i];
                    sum[b] += in_w[q]; sum[64 + b] += w_final[q]; cnt[b] += 1; cnt[64 + b] += 1;
                } else if (ri == REG_MOUTH && net.region[j] == REG_A1) {
                    const int t = net.band[i];
                    sum[128 + t] += in_w[q]; sum[192 + t] += w_final[q]; cnt[128 + t] += 1; cnt[192 + t] += 1;
                }
            }
        }
        std::vector<float> tw(4 * 64, 0.f);
        for (int k = 0; k < 4 * 64; ++k) tw[k] = cnt[k] > 0 ? (float)(sum[k] / cnt[k]) : 0.f;
        write_bin("tone_weights.bin", tw);            // [A1 init 64][A1 final 64][mouth init 64][mouth final 64]
    }
    write_bin("Vs0.bin", Vs0);
    if (full) {
        write_bin("csr_ptr.bin", in_ptr);
        write_bin("csr_eend.bin", in_eend);
        write_bin("csr_pre.bin", in_pre);
        write_bin("csr_meta.bin", in_meta);
        write_bin("csr_w_init.bin", in_w);
        write_bin("csr_w_final.bin", w_final);
    }
    {
        std::ofstream f("params.txt");
        f.precision(9);
        f << "dt " << p.dt << "\nCs_E " << p.Cs_E << "\nCs_I " << p.Cs_I << "\ngLs_E " << p.gLs_E
          << "\ngLs_I " << p.gLs_I << "\nEL " << p.EL << "\nVT " << p.VT << "\ndT_E " << p.dT_E
          << "\ndT_I " << p.dT_I << "\nVr_E " << p.Vr_E << "\nVr_I " << p.Vr_I << "\na_E " << p.a_E
          << "\nb_E " << p.b_E << "\ntauw_E " << p.tauw_E << "\ntref_E " << p.tref_E
          << "\ntref_I " << p.tref_I << "\nV_peak " << p.V_peak << "\nCb " << p.Cb << "\ngLb " << p.gLb
          << "\ng_cb " << p.g_cb << "\nbAP " << p.bAP << "\ninh_soma_frac " << p.inh_soma_frac
          << "\nE_exc " << p.E_exc << "\nE_inh " << p.E_inh << "\ntau_ampa " << p.tau_ampa
          << "\ntau_gaba " << p.tau_gaba << "\ntau_nr " << p.tau_nr << "\ntau_nd " << p.tau_nd
          << "\nnmda_ratio " << p.nmda_ratio << "\nMg " << p.Mg << "\ng_exc " << p.g_exc
          << "\ng_inh " << p.g_inh << "\nlamE_branch_E " << p.lamE_branch_E
          << "\nlamE_branch_I " << p.lamE_branch_I << "\nlamI " << p.lamI << "\nstim_lam " << p.stim_lam
          << "\ntau_plus " << p.tau_plus << "\ntau_minus " << p.tau_minus << "\nA_plus " << p.A_plus
          << "\nA_minus " << p.A_minus << "\nw_max " << p.w_max << "\nL " << L
          << "\nstim_every " << p.stim_every << "\nstim_len " << p.stim_len << "\nwarmup " << p.warmup
          << "\nrec_every " << p.rec_every << "\nw_every " << p.w_every << "\n";
    }
    std::printf("Wrote meta.txt, params.txt and .bin output files%s\n",
                full ? "" : " (synapse lists skipped for N > 20,000)");

    cudaDeviceReset();
    return 0;
}

