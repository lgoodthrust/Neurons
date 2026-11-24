// results.h - statistics and figures for Brain's output files (replaces brain.py's plotting)
#pragma once
#include <string>

// Run statistics + figure (raster, branch/soma voltage, weights); saves test.png.
void plot_run(const std::string& folder, bool show_window = true);
// Sequence task: replay score + figure; saves sequence.png.
void analyze_seq(const std::string& folder, bool show_window = true);
// Frequency task: response to each tone before vs after training; saves frequency.png.
void analyze_tone(const std::string& folder, bool show_window = true);
// Plot whatever the latest output is (sequence run or normal run).
void plot_latest(const std::string& folder);
