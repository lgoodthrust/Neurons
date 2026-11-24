// validate.h - independent CPU reference check of the GPU simulation (replaces brain.py's validator)
#pragma once
#include <string>

// Validate the latest output in `folder` (written by a "validate" run): replays it on the CPU
// in double precision, plus a float rounding-noise baseline, compares, plots (validation.png).
// Returns true on PASS.
bool validate_output(const std::string& folder, bool show_window = true);

// Checks the validator itself on a synthetic network: the correct model must PASS and a
// deliberately broken one (NMDA decay 90 ms instead of 100) must FAIL. No GPU needed.
bool validator_selftest();
