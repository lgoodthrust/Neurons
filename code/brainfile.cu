// brainfile.cu - saving and loading a brain (.brain files)
//
// A .brain file holds everything needed to continue exactly where a brain left off:
// its wiring (incoming synapse lists), the learned synapse strengths, every neuron's
// state, and (for anatomy brains) the neuron positions. Spikes still in flight (the
// last ~3 ms) are not kept. Outgoing lists are rebuilt from the incoming ones on load.
//
// Layout: header, then arrays in this order:
//   in_ptr [N+1] int32, in_eend [N] int32, in_pre [nnz] int32, in_meta [nnz] uint16,
//   in_w [nnz] float, Vs w ref g_gaba x_pre y_post [N] float each,
//   Vb g_ampa n_r n_d [N*B] float each, layout [layout_n] float
#include "brain.cuh"
#include "sim.h"
#include <cstring>
#include <filesystem>

namespace {
const char kMagic[8] = { 'B', 'R', 'A', 'I', 'N', 'F', 'I', 'L' };
const uint32_t kFormat = 1;

struct Header {
    char     magic[8];
    uint32_t format;
    char     version[16];          // BRAIN_VERSION that wrote it
    int32_t  N, N_E, N_A, B, senses, anatomy;
    uint64_t nnz, layout_n;
    double   age_s;                // simulated seconds this brain has lived
    uint32_t reserved[16];         // [0] time step (float bits, 0 = old file: 0.1 ms), [1] synapse setting (0 = old: 160)
                                   // [2] walker creatures, [3] neurons per brain, [4] octopus, [5] rhythm generators
                                   // (all 0 = not a walker file)
};

template <class T> bool put(std::ofstream& f, const std::vector<T>& v) {
    f.write(reinterpret_cast<const char*>(v.data()), (std::streamsize)(v.size() * sizeof(T)));
    return (bool)f;
}
template <class T> bool get(std::ifstream& f, std::vector<T>& v, size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * sizeof(T)));
    return (bool)f;
}

bool read_header(std::ifstream& f, Header& h, std::string& err) {
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f) { err = "file too short"; return false; }
    if (std::memcmp(h.magic, kMagic, 8) != 0) { err = "not a .brain file"; return false; }
    if (h.format != kFormat) { err = "unsupported .brain format " + std::to_string(h.format); return false; }
    if (h.B != B) { err = "brain has " + std::to_string(h.B) + " branches per neuron, this build " + std::to_string(B); return false; }
    if (h.N < 10 || h.N_E <= 0 || h.N_E > h.N) { err = "damaged header"; return false; }
    return true;
}
void header_extras(const Header& h, BrainInfo& info) {
    float dt = 0.f, syn = 0.f;
    std::memcpy(&dt, &h.reserved[0], 4); std::memcpy(&syn, &h.reserved[1], 4);
    info.dt = dt > 0.f ? dt : 0.1f;                       // files from before v9.2
    info.syn = syn > 0.f ? syn : 160.f;
    info.walk_K = (int)h.reserved[2]; info.walk_n = (int)h.reserved[3]; info.walk_mod = (int)h.reserved[4]; info.walk_cpg = (int)h.reserved[5];
}
}  // namespace

bool read_brain_header(const std::string& path, BrainInfo& info, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    Header h{};
    if (!read_header(f, h, err)) return false;
    info.N = h.N; info.N_E = h.N_E; info.N_A = h.N_A; info.senses = h.senses; info.anatomy = h.anatomy;
    info.nnz = (size_t)h.nnz; info.age_s = h.age_s;
    header_extras(h, info);
    return true;
}

bool load_brain(const std::string& path, Network& net, BrainInfo& info, std::string& err) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    Header h{};
    if (!read_header(f, h, err)) return false;
    const size_t N = (size_t)h.N, nnz = (size_t)h.nnz;
    bool ok = get(f, net.in_ptr, N + 1) && get(f, net.in_eend, N) && get(f, net.in_pre, nnz) &&
              get(f, net.in_meta, nnz) && get(f, net.in_w, nnz) &&
              get(f, net.Vs0, N) && get(f, net.st_w, N) && get(f, net.st_ref, N) && get(f, net.st_gaba, N) &&
              get(f, net.st_x, N) && get(f, net.st_y, N) &&
              get(f, net.Vb0, N * B) && get(f, net.st_g_ampa, N * B) && get(f, net.st_n_r, N * B) &&
              get(f, net.st_n_d, N * B) && get(f, net.xyzt, (size_t)h.layout_n);
    if (!ok) { err = "file is truncated or damaged"; return false; }
    // sanity: lists must be consistent before the GPU trusts them
    if (net.in_ptr[0] != 0 || (size_t)net.in_ptr[N] != nnz) { err = "synapse lists are inconsistent"; return false; }
    for (size_t i = 0; i < N; ++i)
        if (net.in_ptr[i] > net.in_ptr[i + 1] || net.in_eend[i] < net.in_ptr[i] || net.in_eend[i] > net.in_ptr[i + 1]) {
            err = "synapse lists are inconsistent"; return false;
        }
    for (size_t q = 0; q < nnz; ++q)
        if (net.in_pre[q] < 0 || net.in_pre[q] >= h.N) { err = "synapse points outside the brain"; return false; }
    net.nnz = nnz;
    finish_network(net, h.N, h.N_E, h.N_A);
    info.N = h.N; info.N_E = h.N_E; info.N_A = h.N_A; info.senses = h.senses; info.anatomy = h.anatomy;
    info.nnz = nnz; info.age_s = h.age_s;
    header_extras(h, info);
    std::printf("Loaded %s: %d neurons, %zu synapses, %.1f min of experience (written by v%.15s), %.1f s\n",
                path.c_str(), h.N, nnz, h.age_s / 60.0, h.version,
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count());
    return true;
}

// Writes a brain held on the host: net.in_* lists and weights, net.Vs0/Vb0 = current voltages,
// net.st_* = the rest of the state, net.xyzt = layout (may be empty).
bool write_brain_file(const std::string& path, const Network& net, int N, int N_E, int N_A, int senses,
                      double age_s, std::string& msg, float dt, float syn, const int* walk) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::error_code ec;
    const std::filesystem::path fp(path);
    if (fp.has_parent_path()) std::filesystem::create_directories(fp.parent_path(), ec);
    Header h{};
    std::memcpy(h.magic, kMagic, 8);
    h.format = kFormat;
    std::strncpy(h.version, BRAIN_VERSION, sizeof(h.version) - 1);
    h.N = N; h.N_E = N_E; h.N_A = N_A; h.B = B; h.senses = senses; h.anatomy = net.xyzt.empty() ? 0 : 1;
    h.nnz = net.in_pre.size(); h.layout_n = net.xyzt.size(); h.age_s = age_s;
    std::memcpy(&h.reserved[0], &dt, 4); std::memcpy(&h.reserved[1], &syn, 4);
    if (walk) { h.reserved[2] = (uint32_t)walk[0]; h.reserved[3] = (uint32_t)walk[1]; h.reserved[4] = (uint32_t)walk[2]; h.reserved[5] = (uint32_t)walk[3]; }

    const std::string tmp = path + ".tmp";               // write to a temp file, then swap it in
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) { msg = "cannot write " + tmp; return false; }
        f.write(reinterpret_cast<const char*>(&h), sizeof(h));
        const bool ok = (bool)f &&
            put(f, net.in_ptr) && put(f, net.in_eend) && put(f, net.in_pre) && put(f, net.in_meta) && put(f, net.in_w) &&
            put(f, net.Vs0) && put(f, net.st_w) && put(f, net.st_ref) && put(f, net.st_gaba) && put(f, net.st_x) &&
            put(f, net.st_y) && put(f, net.Vb0) && put(f, net.st_g_ampa) && put(f, net.st_n_r) && put(f, net.st_n_d) &&
            put(f, net.xyzt);
        f.close();
        if (!ok || !f) { msg = "write failed (disk full?)"; std::filesystem::remove(tmp, ec); return false; }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {                                             // some systems refuse to rename over a file
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp, path, ec);
        if (ec) { msg = "could not replace " + path; return false; }
    }
    const double mb = (double)std::filesystem::file_size(path, ec) / 1048576.0;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "Saved %s (%.0f MB, %.1f min of experience) in %.1f s", path.c_str(), mb,
                  age_s / 60.0, std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count());
    msg = buf;
    std::printf("%s\n", buf);
    return true;
}

// Saves the brain running on the GPU
bool save_brain(const std::string& path, const DevState& s, const Params& p, size_t nnz, int senses,
                const std::vector<float>* layout, double age_s, std::string& msg, const int* walk) {
    const size_t N = (size_t)p.N;
    CUDA_CHECK(cudaDeviceSynchronize());
    Network net;
    net.in_ptr = dev_download(s.in_ptr, N + 1); net.in_eend = dev_download(s.in_eend, N);
    net.in_pre = dev_download(s.in_pre, nnz);   net.in_meta = dev_download(s.in_meta, nnz);
    net.in_w = dev_download(s.in_w, nnz);
    net.Vs0 = dev_download(s.Vs, N);   net.st_w = dev_download(s.w, N);   net.st_ref = dev_download(s.ref, N);
    net.st_gaba = dev_download(s.g_gaba, N); net.st_x = dev_download(s.x_pre, N); net.st_y = dev_download(s.y_post, N);
    net.Vb0 = dev_download(s.Vb, N * B); net.st_g_ampa = dev_download(s.g_ampa, N * B);
    net.st_n_r = dev_download(s.n_r, N * B); net.st_n_d = dev_download(s.n_d, N * B);
    if (layout) net.xyzt = *layout;
    return write_brain_file(path, net, p.N, p.N_E, p.N_A, senses, age_s, msg, p.dt, p.k_local, walk);
}

// ---- for the menu (sim.h): list saved brains with a one-line summary each ----
std::vector<std::string> list_brains(std::vector<std::string>* summaries, int which) {
    std::vector<std::string> out;
    std::error_code ec;
    if (!std::filesystem::is_directory("brains", ec)) return out;
    for (const auto& e : std::filesystem::directory_iterator("brains", ec)) {
        if (!e.is_regular_file() || e.path().extension() != ".brain") continue;
        if (which) {                                       // 1 = live-view brains only, 2 = walkers only
            BrainInfo bi; std::string err;
            if (!read_brain_header(e.path().string(), bi, err)) continue;
            if ((which == 2) != (bi.walk_K > 0)) continue;
        }
        out.push_back(e.path().string());
    }
    std::sort(out.begin(), out.end());
    if (summaries) {
        summaries->clear();
        for (const auto& path : out) {
            BrainInfo bi; std::string err;
            char buf[256];
            if (read_brain_header(path, bi, err) && bi.walk_K > 0) {
                std::snprintf(buf, sizeof(buf), "%d %s%s walkers x %d neurons, %.1f min of experience, %.0f MB, dt %g ms, %.0f syn",
                              bi.walk_K, bi.walk_mod ? "octopus" : "blob", bi.walk_cpg ? " + rhythm" : "", bi.walk_n, bi.age_s / 60.0,
                              (double)std::filesystem::file_size(path, ec) / 1048576.0, bi.dt, bi.syn);
            } else if (read_brain_header(path, bi, err)) {
                const char* kind = bi.anatomy ? "sound anatomy"
                                 : (bi.senses & SENSE_EYES) ? "eyes" : bi.senses ? "senses" : "plain";
                std::snprintf(buf, sizeof(buf), "%d neurons, %s, %.1f min of experience, %.0f MB, dt %g ms, %.0f syn",
                              bi.N, kind, bi.age_s / 60.0, (double)std::filesystem::file_size(path, ec) / 1048576.0,
                              bi.dt, bi.syn);
            } else {
                std::snprintf(buf, sizeof(buf), "(unreadable: %s)", err.c_str());
            }
            summaries->push_back(buf);
        }
    }
    return out;
}
