// audio.h - ears and mouth for the live view (implemented in audio.cpp, plain C++).
#pragma once

// Ears: hear whatever the default speakers/headphones are playing (WASAPI loopback).
bool ears_start();                      // false if no output device could be opened
void ears_stop();
// Loudness of nb log-spaced frequency bands between fmin and fmax (Hz), each 0..1,
// from the latest ~85 ms of sound. Call once per frame (it also drains the capture buffer).
void ears_bands(float* bands, int nb, float fmin, float fmax);

// Mouth: a small additive synthesizer, nb sine tones log-spaced between fmin and fmax.
bool mouth_start();                     // false if the sound output could not be opened
void mouth_stop();
// amp[b] = loudness of tone b (0..1), volume = master volume (0..1). Call once per frame.
void mouth_update(const float* amp, int nb, float fmin, float fmax, float volume);