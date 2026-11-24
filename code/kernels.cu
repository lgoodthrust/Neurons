// kernels.cu - the GPU simulation: neuron update, spike delivery, STDP, main kernel
#include "brain.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned long long gtime() {      // nanoseconds
    unsigned long long t;
#ifdef __CUDA_ARCH__
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
#else
    t = 0;
#endif
    return t;
}

__device__ __forceinline__ float mg_block(float V, float mg_c) {     // mg_c = Mg / 3.57
    return 1.0f / (1.0f + mg_c * __expf(-0.062f * V));
}

// Exact Poisson sample by inverse CDF from one uniform u in [0,1).
__device__ __forceinline__ float poisson_inv(float u, float lam, float exp_neg_lam) {
    float p = exp_neg_lam, cdf = p;
    int k = 0;
    while (u >= cdf && k < 64) {
        ++k;
        p *= lam / (float)k;
        cdf += p;
    }
    return (float)k;
}
__device__ __forceinline__ float u01(unsigned int x) {
    return (float)(x >> 8) * (1.0f / 16777216.0f);
}


// -----------------------------------------------------------------------------
//  Phase 1: advance neuron n by M steps (no communication)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void integrate_neuron(const DevState& s, const Params& p,
                                                 const int n, const int e0, const int par) {
    const int  N = p.N, N_E = p.N_E, N_A = p.N_A, M = p.M;
    const bool isE    = n < N_E;
    const bool groupA = n < N_A;
    const float dtCs   = isE ? p.dtCs_E : p.dtCs_I;      // dt / C_soma
    const float gLs    = isE ? p.gLs_E : p.gLs_I;
    const float dT     = isE ? p.dT_E  : p.dT_I;
    const float inv_dT = isE ? p.inv_dT_E : p.inv_dT_I;
    const float Vr     = isE ? p.Vr_E  : p.Vr_I;
    const float a      = isE ? p.a_E   : 0.0f;
    const float bw     = isE ? p.b_E   : 0.0f;
    const float dtTauw = isE ? p.dtTauw_E : p.dt;       // dt / tau_w (I cells: a = 0, w stays 0)
    const float t_ref  = isE ? p.tref_E : p.tref_I;
    const float frac_s = isE ? p.inh_soma_frac : 1.0f;
    const float lamE   = isE ? p.lamE_branch_E : p.lamE_branch_I;
    const float expE   = isE ? p.expE_E : p.expE_I;
    const int   tslot  = n == 0 ? 0 : n == N_A ? 1 : n == N_E ? 2 : -1;

    float Vs = s.Vs[n], w = s.w[n], ref = s.ref[n], g_gaba = s.g_gaba[n];
    float x_pre = s.x_pre[n], y_post = s.y_post[n];
    float Vb[B], g_ampa[B], n_r[B], n_d[B];
    static_assert(B == 4, "float4 loads assume 4 branches");
    {
        const float4 a = reinterpret_cast<const float4*>(s.Vb)[n];
        const float4 b = reinterpret_cast<const float4*>(s.g_ampa)[n];
        const float4 c = reinterpret_cast<const float4*>(s.n_r)[n];
        const float4 d = reinterpret_cast<const float4*>(s.n_d)[n];
        Vb[0] = a.x; Vb[1] = a.y; Vb[2] = a.z; Vb[3] = a.w;
        g_ampa[0] = b.x; g_ampa[1] = b.y; g_ampa[2] = b.z; g_ampa[3] = b.w;
        n_r[0] = c.x; n_r[1] = c.y; n_r[2] = c.z; n_r[3] = c.w;
        n_d[0] = d.x; n_d[1] = d.y; n_d[2] = d.z; n_d[3] = d.w;
    }
    bool spiked_since_sample = s.tflag[n] != 0;
    const int poke_until = s.poke_end[n];
    // eyes: this neuron's pixel brightness -> extra Poisson input on branch 0
    const float eye_lam = (n < p.eye_n) ? s.eye[n] * p.eye_gain : 0.0f;
    const float eye_exp = __expf(-eye_lam);

    s.x_start[(size_t)par * N + n] = x_pre;
    s.y_start[n] = y_post;
    unsigned int ownb = 0;

    // prefetch the first step's inputs (all slots of this epoch are already final)
    float in_e[B], in_i;
    {
        const int slot0 = e0 & (L - 1);
        const float4 v = reinterpret_cast<const float4*>(s.buf_exc)[(size_t)slot0 * N + n];
        in_e[0] = v.x; in_e[1] = v.y; in_e[2] = v.z; in_e[3] = v.w;
        in_i = s.buf_inh[(size_t)slot0 * N + n];
    }

    for (int u = 0; u < M; ++u) {
        const int  step = e0 + u;
        const int  slot = step & (L - 1);
        bool stim;
        if (p.seq) {                                          // sequence task schedule
            stim = false;
            if (n < p.seq_K * p.seq_size) {
                const int g  = n / p.seq_size;
                const int ph = step % p.seq_cycle;
                if (step >= p.train0 && step < p.train1)      // training: A, B, C, D in order
                    stim = ph >= g * p.seq_delta && ph < g * p.seq_delta + p.seq_len;
                else if (p.seq_test && step >= p.test0) {     // test: cue A or D alone
                    const int cue = ((step / p.seq_cycle) & 1) ? p.seq_K - 1 : 0;
                    stim = g == cue && ph < p.seq_len;
                }
            }
            stim = stim || step < poke_until;
        } else {
            stim = (groupA && p.auto_stim && (step % p.stim_every) < p.stim_len) || step < poke_until;
        }

        // start loading the NEXT step's inputs now; they arrive while this step computes
        float nx_e[B], nx_i = 0.0f;
        if (u + 1 < M) {
            const int nslot = (step + 1) & (L - 1);
            const float4 v = reinterpret_cast<const float4*>(s.buf_exc)[(size_t)nslot * N + n];
            nx_e[0] = v.x; nx_e[1] = v.y; nx_e[2] = v.z; nx_e[3] = v.w;
            nx_i = s.buf_inh[(size_t)nslot * N + n];
        } else {
            #pragma unroll
            for (int k = 0; k < B; ++k) nx_e[k] = 0.0f;
        }

        float uu[8];
        if (!p.deterministic) {
            // counter-based Philox: random numbers computed from (neuron, step), no stored state
            const uint2 key = make_uint2(0x2545F491u, 0x9E3779B9u);
            const uint4 r0 = curand_Philox4x32_10(make_uint4((unsigned)step, (unsigned)n, 0u, 0x5EEDu), key);
            const uint4 r1 = curand_Philox4x32_10(make_uint4((unsigned)step, (unsigned)n, 1u, 0x5EEDu), key);
            uu[0] = u01(r0.x); uu[1] = u01(r0.y); uu[2] = u01(r0.z); uu[3] = u01(r0.w);
            uu[4] = u01(r1.x); uu[5] = u01(r1.y); uu[6] = u01(r1.z); uu[7] = u01(r1.w);
        }
        
        #pragma unroll
        for (int k = 0; k < B; ++k) {
            float ext = p.deterministic ? lamE : poisson_inv(uu[k], lamE, expE);
            if (stim && k == 0)
                ext += p.deterministic ? p.stim_lam : poisson_inv(uu[5], p.stim_lam, p.expStim);
            if (k == 0 && eye_lam > 0.0f) ext += poisson_inv(uu[6], eye_lam, eye_exp);
            const float in = in_e[k] + ext * p.g_exc;
            g_ampa[k] += in;
            n_r[k]    += in * p.nmda_ratio;
            n_d[k]    += in * p.nmda_ratio;
        }
        reinterpret_cast<float4*>(s.buf_exc)[(size_t)slot * N + n] = make_float4(0.f, 0.f, 0.f, 0.f);
        const float ext_i = p.deterministic ? p.lamI : poisson_inv(uu[4], p.lamI, p.expI);
        g_gaba += in_i + ext_i * (p.g_inh * p.inh_scale);
        s.buf_inh[(size_t)slot * N + n] = 0.0f;

        float I_soma = frac_s * g_gaba * (p.E_inh - Vs);
        float dVb[B];
        #pragma unroll
        for (int k = 0; k < B; ++k) {
            const float Vsite = isE ? Vb[k] : Vs;
            const float gn    = (n_d[k] - n_r[k]) * mg_block(Vsite, p.mg_c);
            const float Iexc  = (g_ampa[k] + gn) * (p.E_exc - Vsite);
            if (isE) {
                const float Icpl  = p.g_cb * (Vb[k] - Vs);
                const float Iinhb = (1.0f - frac_s) * g_gaba * (1.0f / B) * (p.E_inh - Vb[k]);   // 1/B exact (B = 4)
                I_soma += Icpl;
                dVb[k] = (-p.gLb * (Vb[k] - p.EL) - Icpl + Iexc + Iinhb) * p.dtCb;
            } else {
                I_soma += Iexc;
                dVb[k] = 0.0f;
            }
        }
        const float expt = gLs * dT * __expf(fminf((Vs - p.VT) * inv_dT, 20.0f));
        const float dVs  = (-gLs * (Vs - p.EL) + expt - w + I_soma) * dtCs;
        const float dw   = (a * (Vs - p.EL) - w) * dtTauw;

        if (ref <= 0.0f) Vs += dVs;
        #pragma unroll
        for (int k = 0; k < B; ++k) Vb[k] = isE ? Vb[k] + dVb[k] : Vs;
        w   += dw;
        ref -= p.dt;

        #pragma unroll
        for (int k = 0; k < B; ++k) {
            g_ampa[k] -= g_ampa[k] * p.k_ampa;
            n_r[k]    -= n_r[k] * p.k_nr;
            n_d[k]    -= n_d[k] * p.k_nd;
        }
        g_gaba -= g_gaba * p.k_gaba;
        x_pre  -= x_pre * p.k_plus;
        y_post -= y_post * p.k_minus;

        const bool spk = Vs >= p.V_peak;
        if (spk) {
            ownb |= 1u << u;
            if (ownb == (1u << u)) {                       // first spike of this epoch: join the list
                const int k = atomicAdd(&s.ep_cnt[par], 1);
                if (k < N) s.ep_id[(size_t)par * N + k] = n;
            }
            s.last_spike[n] = step;
            if (s.spike_n) s.spike_n[n] += 1;          // live view: exact spike counts for the graphs
            if (p.spike_cap > 0) {
                const unsigned long long idx = atomicAdd(s.spike_count, 1ULL);
                if (idx < (unsigned long long)p.spike_cap) { s.spike_step[idx] = step; s.spike_id[idx] = n; }
            }
            x_pre  += 1.0f;
            y_post += 1.0f;
            Vs = Vr;
            if (isE) {
                #pragma unroll
                for (int k = 0; k < B; ++k) Vb[k] += p.bAP;
            }
            w  += bw;
            ref = t_ref;
        }

        if (tslot >= 0 && p.record) {
            spiked_since_sample |= spk;
            if (step % p.rec_every == 0) {
                const size_t r = (size_t)(step / p.rec_every) * NT + tslot;
                s.tr_s[r] = spiked_since_sample ? p.V_peak : Vs;
                #pragma unroll
                for (int k = 0; k < B; ++k) s.tr_b[r * B + k] = Vb[k];
                spiked_since_sample = false;
            }
        }

        #pragma unroll
        for (int k = 0; k < B; ++k) in_e[k] = nx_e[k];     // the prefetched inputs become current
        in_i = nx_i;
    }

    s.ep_bits[(size_t)par * N + n] = (uint16_t)ownb;
    s.Vs[n] = Vs; s.w[n] = w; s.ref[n] = ref; s.g_gaba[n] = g_gaba;
    s.x_pre[n] = x_pre; s.y_post[n] = y_post;
    reinterpret_cast<float4*>(s.Vb)[n]     = make_float4(Vb[0], Vb[1], Vb[2], Vb[3]);
    reinterpret_cast<float4*>(s.g_ampa)[n] = make_float4(g_ampa[0], g_ampa[1], g_ampa[2], g_ampa[3]);
    reinterpret_cast<float4*>(s.n_r)[n]    = make_float4(n_r[0], n_r[1], n_r[2], n_r[3]);
    reinterpret_cast<float4*>(s.n_d)[n]    = make_float4(n_d[0], n_d[1], n_d[2], n_d[3]);
#pragma unroll
    for (int k = 0; k < B; ++k) {
        const size_t o = (size_t)n * B + k;
        if (e0 >= p.warmup) {                          // statistics: one sample per epoch (1 ms)
            s.vb_sum[o]  += Vb[k];
            s.plateau[o] += (Vb[k] > -35.0f) ? 1.0f : 0.0f;
        }
    }
    s.tflag[n] = spiked_since_sample ? 1 : 0;
}

// Reward learning: add an STDP change to synapse q's eligibility trace (decayed to step t first)
__device__ __forceinline__ void elig_add(const DevState& s, const Params& p, int q, float d, int t) {
    const float e = s.elig[q] * __expf(-(float)(t - s.elig_t[q]) * p.k_elig) + d;
    s.elig[q] = e; s.elig_t[q] = t;
}

// Reward (dopamine): turn each plastic synapse's recent eligibility into a real weight change.
// One thread per excitatory neuron, over its plastic (E) inputs.
__global__ void reward_kernel(const int* __restrict__ in_ptr, const int* __restrict__ in_eend,
                              float* __restrict__ in_w, const float* __restrict__ elig,
                              const int* __restrict__ elig_t, int N_E, int now, float k_elig,
                              float amount, float w_max) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N_E) return;
    for (int q = in_ptr[n]; q < in_eend[n]; ++q) {
        const float e = elig[q] * __expf(-(float)(now - elig_t[q]) * k_elig);
        if (e != 0.0f) in_w[q] = fminf(fmaxf(in_w[q] + amount * e, 0.0f), w_max);
    }
}

// -----------------------------------------------------------------------------
//  Phase 3 helper: one presynaptic spike arriving at neuron n (CSR entry q)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void receive(const DevState& s, const Params& p, const int n,
                                        const int q, const int t, const float y, const bool plastic) {
    const float w = s.in_w[q];
    if (w == 0.0f) return;
    const int      j    = __ldg(&s.in_pre[q]);
    const unsigned meta = __ldg(&s.in_meta[q]);
    const int      arr  = (t + (int)(meta & 63u)) & (L - 1);
    if (j < p.N_E) s.buf_exc[((size_t)arr * p.N + n) * B + (meta >> 6)] += w;
    else           s.buf_inh[(size_t)arr * p.N + n] += w * p.inh_scale;
    if (plastic) {                                                // pre after post: depression
        if (p.da_mode) elig_add(s, p, q, -p.A_minus * y, t);
        else           s.in_w[q] = fmaxf(w - p.A_minus * y, 0.0f);
    }
}

// Same as receive(), but everything except the weight comes from the note, and the weight
// (the synapse's current value) is supplied by the caller, which may have prefetched it.
__device__ __forceinline__ void receive_note(const DevState& s, const Params& p, const int n,
                                             const int q, const unsigned meta, const int t,
                                             const float y, const bool plastic, const float w) {
    if (w == 0.0f) return;
    const int arr = (t + (int)(meta & 63u)) & (L - 1);
    if (meta & 0x100u) s.buf_exc[((size_t)arr * p.N + n) * B + ((meta >> 6) & 3u)] += w;
    else               s.buf_inh[(size_t)arr * p.N + n] += w * p.inh_scale;
    if (plastic) {                                                // pre after post: depression
        if (p.da_mode) elig_add(s, p, q, -p.A_minus * y, t);
        else           s.in_w[q] = fmaxf(w - p.A_minus * y, 0.0f);
    }
}

// -----------------------------------------------------------------------------
//  Main kernel (cooperative launch). Runs steps [step_begin, step_end), both
//  multiples of p.M. Thread g owns neurons g, g + T, g + 2T, ... (T = all threads).
// -----------------------------------------------------------------------------
__global__ void __launch_bounds__(MAX_THREADS, MIN_BLOCKS)
simulate(DevState s, Params p, int step_begin, int step_end) {
    cg::grid_group grid = cg::this_grid();
    const int N = p.N, N_E = p.N_E, N_A = p.N_A, M = p.M;
    const int gtid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;
    const int R     = (N + total - 1) / total;           // neurons per thread (rounds)
    const unsigned lane = threadIdx.x & 31u;
    const bool timer = gtid == 0;
    unsigned long long tI = 0, tS = 0, tD = 0;

    for (int e0 = step_begin; e0 < step_end; e0 += M) {
        const int par = (e0 / M) & 1;
        const unsigned long long c0 = gtime();

        // ======================= 1. INTEGRATE =======================
        for (int r = 0; r < R; ++r) {
            const int n = r * total + gtid;
            if (n < N) integrate_neuron(s, p, n, e0, par);
        }
        const unsigned long long c1 = gtime();

        // ======================= 2. GRID-WIDE BARRIER =======================
        grid.sync();
        const unsigned long long c2 = gtime();

        // ======================= 3a. PUSH: leave notes with targets ==========
        {
            const int cnt    = min(s.ep_cnt[par], N);
            const int warp_g = gtid >> 5, nwarps = total >> 5;
            for (int k = warp_g; k < cnt; k += nwarps) {         // one warp per spiking neuron
                const int      j  = s.ep_id[(size_t)par * N + k];
                const unsigned jb = s.ep_bits[(size_t)par * N + j];      // when j spiked this epoch
                const int oa = __ldg(&s.out_ptr[j]), ob = __ldg(&s.out_ptr[j + 1]);
                for (int o = oa + (int)lane; o < ob; o += 32) {
                    const int      tgt  = __ldg(&s.out_post[o]);
                    const unsigned q    = (unsigned)__ldg(&s.out_q[o]);
                    const unsigned meta = __ldg(&s.out_meta[o]);
                    const int      slot = atomicAdd(&s.hitcnt[tgt], 1);
                    if (slot < MAXHIT) s.hitn[(size_t)tgt * MAXHIT + slot] = make_uint2(q, jb | (meta << 16));
                }
            }
        }
        const unsigned long long c3 = gtime();
        grid.sync();
        const unsigned long long c4 = gtime();
        if (timer) s.ep_cnt[par] = 0;                        // list fully consumed: reuse in 2 epochs

        // ======================= 3b. PULL: deliver + STDP ====================
        const uint16_t* bits = s.ep_bits + (size_t)par * N;
        const float*    xs   = s.x_start + (size_t)par * N;
        for (int r = 0; r < R; ++r) {                    // uniform across the warp (warp ops inside)
            const int  n      = r * total + gtid;
            const bool alive  = n < N;
            const bool isE    = alive && n < N_E;
            const unsigned ownb = alive ? bits[n] : 0u;
            float y = alive ? s.y_start[n] : 0.0f;
            int q0 = 0, qe = 0, q1 = 0;
            if (alive) { q0 = __ldg(&s.in_ptr[n]); qe = __ldg(&s.in_eend[n]); q1 = __ldg(&s.in_ptr[n + 1]); }

            // my notes from the push step: each carries synapse index, spike bits and delay/branch/type
            const int    hc       = alive ? s.hitcnt[n] : 0;
            const bool   overflow = hc > MAXHIT;
            const int    nh       = overflow ? 0 : hc;
            const uint2* notes    = s.hitn + (size_t)n * MAXHIT;

            // v9.6 WARP POOL. A neuron's synapses can only change mid-epoch through potentiation, which
            // needs the neuron itself to spike this epoch as an E cell (~1 epoch in 600). All other
            // neurons' notes for the whole epoch go into one pool that the warp's 32 lanes work through
            // together (32 memory trips in flight instead of 1). Each note's deliveries still run in
            // step order in one lane, with this neuron's trace replayed exactly as the step loop does.
            // The rare spiking E neurons (and overflows) keep the step-by-step path below.
            const bool slow = alive && (overflow || (isE && ownb != 0u));
            {
                const int mine = (alive && !slow) ? nh : 0;          // notes this lane puts in the pool
                int incl = mine;                                     // inclusive prefix sum over the warp
                #pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    const int v = __shfl_up_sync(0xffffffffu, incl, d);
                    if ((int)lane >= d) incl += v;
                }
                const int excl  = incl - mine;
                const int total_notes = __shfl_sync(0xffffffffu, incl, 31);
                for (int base = 0; base < total_notes; base += 32) {    // uniform across the warp
                    const int g = base + (int)lane;
                    int o = 0;                                       // owner: last lane with excl <= g
                    #pragma unroll
                    for (int st = 16; st > 0; st >>= 1) {
                        const int e = __shfl_sync(0xffffffffu, excl, o + st);
                        if (e <= g) o += st;
                    }
                    const int   eo  = __shfl_sync(0xffffffffu, excl, o);
                    const float y0  = __shfl_sync(0xffffffffu, y, o);
                    const int   qeo = __shfl_sync(0xffffffffu, qe, o);
                    if (g < total_notes) {
                        const int   no   = n - (int)lane + o;            // the owner's neuron
                        const uint2 nt   = s.hitn[(size_t)no * MAXHIT + (g - eo)];
                        const int   q    = (int)nt.x;
                        const unsigned meta = nt.y >> 16;
                        const bool  plastic = no < N_E && q < qeo;
                        float w  = s.in_w[q];
                        float yv = y0;
                        int   uu = 0;
                        unsigned b = nt.y & 0xFFFFu;
                        while (b) {                                  // this note's spikes, in step order
                            const int u = __ffs(b) - 1;
                            b &= b - 1;
                            for (; uu <= u; ++uu) yv -= yv * p.k_minus;   // trace at step u (no own spikes)
                            if (w == 0.0f) break;                    // as receive_note: nothing to deliver
                            const int t   = e0 + u;
                            const int arr = (t + (int)(meta & 63u)) & (L - 1);
                            if (meta & 0x100u) atomicAdd(&s.buf_exc[((size_t)arr * N + no) * B + ((meta >> 6) & 3u)], w);
                            else               atomicAdd(&s.buf_inh[(size_t)arr * N + no], w * p.inh_scale);
                            if (plastic) {                           // pre after post: depression
                                if (p.da_mode) elig_add(s, p, q, -p.A_minus * yv, t);
                                else { w = fmaxf(w - p.A_minus * yv, 0.0f); s.in_w[q] = w; }
                            }
                        }
                    }
                }
            }

            // step-by-step path (spiking E neurons): prefetch the first two notes' weights (v9.5)
            const int nhs = slow ? nh : 0;
            float pw0 = 0.0f, pw1 = 0.0f;
            if (nhs > 0) pw0 = s.in_w[notes[0].x];
            if (nhs > 1) pw1 = s.in_w[notes[1].x];
            unsigned any = 0;                                // steps in which any of my (slow) inputs spiked
            for (int k = 0; k < nhs; ++k) any |= notes[k].y & 0xFFFFu;
            if (alive) s.hitcnt[n] = 0;                      // ready for the next epoch's notes

            for (int u = 0; u < M; ++u) {
                const int t = e0 + u;
                y -= y * p.k_minus;

                // presynaptic spikes at step t: deliver + depression
                if (alive) {
                    if (!overflow) {
                        if ((any >> u) & 1u) {
                            for (int k = 0; k < nhs; ++k) {
                                const uint2 nt = notes[k];
                                if ((nt.y >> u) & 1u) {
                                    const int  q       = (int)nt.x;
                                    const bool plastic = isE && q < qe;
                                    // The prefetched weight is still exact unless this epoch already
                                    // changed it: only plain STDP changes weights mid-epoch, through
                                    // this neuron's own earlier spikes (potentiation) or an earlier
                                    // delivery of this same note (depression). Then: reload.
                                    const bool fresh = !plastic || p.da_mode ||
                                        ((ownb & ((1u << u) - 1u)) == 0u && (nt.y & ((1u << u) - 1u)) == 0u);
                                    const float w = (fresh && k < 2) ? (k == 0 ? pw0 : pw1) : s.in_w[q];
                                    receive_note(s, p, n, q, nt.y >> 16, t, y, plastic, w);
                                }
                            }
                        }
                    } else {                              // rare: rescan the whole list
                        for (int q = q0; q < q1; ++q)
                            if ((bits[s.in_pre[q]] >> u) & 1u) receive(s, p, n, q, t, y, isE && q < qe);
                    }
                }

                // own spikes at step t: potentiation, shared by the warp
                __syncwarp();
                unsigned spiking = __ballot_sync(0xffffffffu, isE && ((ownb >> u) & 1u));
                while (spiking) {
                    const int src  = __ffs(spiking) - 1;
                    spiking &= spiking - 1;
                    const int post = n - (int)lane + src;     // lanes own consecutive neurons
                    const int pa = __ldg(&s.in_ptr[post]), pe = __ldg(&s.in_eend[post]);
                    for (int q = pa + (int)lane; q < pe; q += 32) {
                        const int      j  = __ldg(&s.in_pre[q]);
                        float          xj = xs[j];
                        const unsigned bj = bits[j];
                        for (int v = 0; v <= u; ++v) {
                            xj -= xj * p.k_plus;
                            if (v < u && ((bj >> v) & 1u)) xj += 1.0f;
                        }
                        if (p.da_mode) elig_add(s, p, q, p.A_plus * xj, t);
                        else           s.in_w[q] = fminf(s.in_w[q] + p.A_plus * xj, p.w_max);
                    }
                }
                __syncwarp();
                if ((ownb >> u) & 1u) y += 1.0f;
            }

            if (isE && p.record && e0 % p.w_every == 0) { // summarize incoming E->E weights
                float sA = 0.0f, sO = 0.0f;
                const bool groupA = n < N_A;
                for (int q = q0; q < qe; ++q) {
                    const float wv = s.in_w[q];
                    if (groupA && s.in_pre[q] < N_A) sA += wv; else sO += wv;
                }
                const size_t o = (size_t)(e0 / p.w_every) * N + n;
                s.snapA[o] = sA;
                s.snapO[o] = sO;
                if (p.seq && n < p.seq_K * p.seq_size) {          // weights between task groups
                    const int gp = n / p.seq_size, K = p.seq_K;
                    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (int q = q0; q < qe; ++q) {
                        const int j = s.in_pre[q];
                        if (j < K * p.seq_size) acc[j / p.seq_size] += s.in_w[q];
                    }
                    float* dst = s.snapG + (size_t)(e0 / p.w_every) * K * K;
                    for (int gq = 0; gq < K; ++gq)
                        if (acc[gq] != 0.0f) atomicAdd(&dst[gq * K + gp], acc[gq]);
                }
            }
        }
        const unsigned long long c5 = gtime();
        tI += c1 - c0;
        tS += (c2 - c1) + (c4 - c3);
        tD += (c3 - c2) + (c5 - c4);
    }
    if (timer) { s.prof[0] += tI; s.prof[1] += tS; s.prof[2] += tD; }
}

