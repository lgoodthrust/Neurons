// audio.cpp - ears (WASAPI loopback capture of what the speakers play) and mouth
// (a small additive synthesizer played through waveOut).
// Plain C++, compiled by the host compiler: the Windows audio headers stay out of the CUDA files.
#include "audio.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <mmsystem.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "winmm.lib")
#ifndef WAVE_FORMAT_IEEE_FLOAT
#define WAVE_FORMAT_IEEE_FLOAT 0x0003
#endif
#ifndef WAVE_FORMAT_EXTENSIBLE
#define WAVE_FORMAT_EXTENSIBLE 0xFFFE
#endif

namespace {
    // ------------------------------- ears -------------------------------
    const int FFT_N = 4096, RING = 8192;                 // ~85 ms window, ~12 Hz resolution; RING must be a power of two
    IAudioClient* g_ac = nullptr;
    IAudioCaptureClient* g_cc = nullptr;
    bool  g_com = false, g_float = true;
    int   g_ch = 2, g_bits = 32, g_rate = 48000;
    std::vector<float> g_ring(RING, 0.0f), g_prev;
    size_t g_rpos = 0;

    void pull_audio() {
        if (!g_cc) return;
        UINT32 packet = 0;
        while (SUCCEEDED(g_cc->GetNextPacketSize(&packet)) && packet > 0) {
            BYTE* data = nullptr; UINT32 frames = 0; DWORD flags = 0;
            if (FAILED(g_cc->GetBuffer(&data, &frames, &flags, nullptr, nullptr))) break;
            const bool silent = (flags & AUDCLNT_BUFFERFLAGS_SILENT) != 0;
            for (UINT32 i = 0; i < frames; ++i) {
                float v = 0.0f;
                if (!silent) {
                    for (int c = 0; c < g_ch; ++c) {
                        if (g_float)          v += reinterpret_cast<const float*>(data)[(size_t)i * g_ch + c];
                        else if (g_bits == 16) v += reinterpret_cast<const short*>(data)[(size_t)i * g_ch + c] / 32768.0f;
                    }
                    v /= (float)g_ch;                      // mix down to mono
                }
                g_ring[g_rpos++ & (RING - 1)] = v;
            }
            g_cc->ReleaseBuffer(frames);
        }
    }

    void fft(std::vector<std::complex<float>>& a) {      // in-place radix-2
        const int n = (int)a.size();
        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            const float ang = -6.28318530718f / len;
            const std::complex<float> wl(std::cos(ang), std::sin(ang));
            for (int i = 0; i < n; i += len) {
                std::complex<float> w(1.0f, 0.0f);
                for (int k = 0; k < len / 2; ++k) {
                    const std::complex<float> u = a[i + k], v = a[i + k + len / 2] * w;
                    a[i + k] = u + v; a[i + k + len / 2] = u - v;
                    w *= wl;
                }
            }
        }
    }

    // ------------------------------- mouth -------------------------------
    const int MB = 4, MLEN = 1024, MRATE = 48000;        // 4 blocks of ~21 ms queued (~85 ms latency)
    HWAVEOUT g_wo = nullptr;
    WAVEHDR  g_hdr[MB];
    std::vector<short> g_buf[MB];
    bool g_queued[MB];
    std::vector<float> m_phase, m_amp;
}  // namespace

bool ears_start() {
    const HRESULT hc = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    g_com = SUCCEEDED(hc);                           // RPC_E_CHANGED_MODE: COM already set up, still usable
    IMMDeviceEnumerator* en = nullptr;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator), (void**)&en))) return false;
    IMMDevice* dev = nullptr;
    const HRESULT hd = en->GetDefaultAudioEndpoint(eRender, eConsole, &dev);   // what the speakers play
    en->Release();
    if (FAILED(hd)) return false;
    const HRESULT ha = dev->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&g_ac);
    dev->Release();
    if (FAILED(ha)) return false;
    WAVEFORMATEX* wf = nullptr;
    if (FAILED(g_ac->GetMixFormat(&wf))) { ears_stop(); return false; }
    g_ch = wf->nChannels; g_bits = wf->wBitsPerSample; g_rate = (int)wf->nSamplesPerSec;
    g_float = wf->wFormatTag == WAVE_FORMAT_IEEE_FLOAT || (wf->wFormatTag == WAVE_FORMAT_EXTENSIBLE && g_bits == 32);
    const HRESULT hi = g_ac->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK,
        2000000 /* 200 ms buffer */, 0, wf, nullptr);
    CoTaskMemFree(wf);
    if (FAILED(hi) || FAILED(g_ac->GetService(__uuidof(IAudioCaptureClient), (void**)&g_cc))) { ears_stop(); return false; }
    if (FAILED(g_ac->Start())) { ears_stop(); return false; }
    return true;
}

void ears_stop() {
    if (g_ac) g_ac->Stop();
    if (g_cc) { g_cc->Release(); g_cc = nullptr; }
    if (g_ac) { g_ac->Release(); g_ac = nullptr; }
    if (g_com) { CoUninitialize(); g_com = false; }
}

void ears_bands(float* bands, int nb, float fmin, float fmax) {
    pull_audio();
    if ((int)g_prev.size() != nb) g_prev.assign(nb, 0.0f);
    static std::vector<std::complex<float>> a(FFT_N);
    for (int i = 0; i < FFT_N; ++i) {                // newest FFT_N samples, Hann window
        const float hann = 0.5f - 0.5f * std::cos(6.28318530718f * i / (FFT_N - 1));
        a[i] = std::complex<float>(g_ring[(g_rpos - FFT_N + i) & (RING - 1)] * hann, 0.0f);
    }
    fft(a);
    const float res = (float)g_rate / FFT_N;         // Hz per FFT bin
    const float full = (FFT_N / 4.0f) * (FFT_N / 4.0f);   // |X|^2 of a full-scale sine
    for (int b = 0; b < nb; ++b) {
        const float f0 = fmin * std::pow(fmax / fmin, (float)b / nb);
        const float f1 = fmin * std::pow(fmax / fmin, (float)(b + 1) / nb);
        const int k0 = std::max(1, (int)(f0 / res));
        const int k1 = std::min(FFT_N / 2, std::max(k0 + 1, (int)std::ceil(f1 / res)));
        float e = 0.0f;
        for (int k = k0; k < k1; ++k) e += std::norm(a[k]);
        e /= (float)(k1 - k0) * full;
        const float db = 10.0f * std::log10(e + 1e-12f);              // 0 dB = full scale
        const float v = std::min(1.0f, std::max(0.0f, (db + 60.0f) / 60.0f));   // -60..0 dB -> 0..1
        g_prev[b] = std::max(v, g_prev[b] * 0.85f);                   // fast attack, ~0.1 s release
        bands[b] = g_prev[b];
    }
}

bool mouth_start() {
    WAVEFORMATEX f{};
    f.wFormatTag = WAVE_FORMAT_PCM; f.nChannels = 1; f.nSamplesPerSec = MRATE; f.wBitsPerSample = 16;
    f.nBlockAlign = 2; f.nAvgBytesPerSec = MRATE * 2;
    if (waveOutOpen(&g_wo, WAVE_MAPPER, &f, 0, 0, CALLBACK_NULL) != MMSYSERR_NOERROR) { g_wo = nullptr; return false; }
    for (int b = 0; b < MB; ++b) {
        g_buf[b].assign(MLEN, 0);
        std::memset(&g_hdr[b], 0, sizeof(WAVEHDR));
        g_hdr[b].lpData = (LPSTR)g_buf[b].data();
        g_hdr[b].dwBufferLength = MLEN * sizeof(short);
        waveOutPrepareHeader(g_wo, &g_hdr[b], sizeof(WAVEHDR));
        g_queued[b] = false;
    }
    return true;
}

void mouth_stop() {
    if (!g_wo) return;
    waveOutReset(g_wo);
    for (int b = 0; b < MB; ++b) waveOutUnprepareHeader(g_wo, &g_hdr[b], sizeof(WAVEHDR));
    waveOutClose(g_wo);
    g_wo = nullptr;
}

void mouth_update(const float* amp, int nb, float fmin, float fmax, float volume) {
    if (!g_wo) return;
    if ((int)m_phase.size() != nb) { m_phase.assign(nb, 0.0f); m_amp.assign(nb, 0.0f); }
    for (int blk = 0; blk < MB; ++blk) {
        if (g_queued[blk] && !(g_hdr[blk].dwFlags & WHDR_DONE)) continue;    // still playing
        static std::vector<float> mix(MLEN);
        std::fill(mix.begin(), mix.end(), 0.0f);
        for (int b = 0; b < nb; ++b) {
            const float a0 = m_amp[b], a1 = amp[b];                          // ramp: no clicks
            if (a0 < 1e-4f && a1 < 1e-4f) { m_amp[b] = a1; continue; }
            const float f = fmin * std::pow(fmax / fmin, (b + 0.5f) / nb);
            const float dph = 6.28318530718f * f / MRATE;
            float ph = m_phase[b];
            for (int i = 0; i < MLEN; ++i) {
                mix[i] += (a0 + (a1 - a0) * i / MLEN) * std::sin(ph);
                ph += dph;
            }
            m_phase[b] = std::fmod(ph, 6.28318530718f);
            m_amp[b] = a1;
        }
        const float g = volume / std::sqrt((float)nb) * 2.0f;
        for (int i = 0; i < MLEN; ++i)
            g_buf[blk][i] = (short)(32000.0f * std::tanh(mix[i] * g));      // soft clip
        g_hdr[blk].dwFlags &= ~WHDR_DONE;
        waveOutWrite(g_wo, &g_hdr[blk], sizeof(WAVEHDR));
        g_queued[blk] = true;
    }
}

#else   // not Windows: no ears or mouth
bool ears_start() { return false; }
void ears_stop() {}
void ears_bands(float* bands, int nb, float, float) { for (int b = 0; b < nb; ++b) bands[b] = 0.0f; }
bool mouth_start() { return false; }
void mouth_stop() {}
void mouth_update(const float*, int, float, float, float) {}
#endif