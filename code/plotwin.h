// plotwin.h - a window for result figures (Dear ImGui + ImPlot), saved as PNG.
#pragma once
#include <functional>
#include <string>

// Opens a window and calls draw() every frame inside a full-window ImGui panel
// (ImPlot is ready to use). The first complete frame is saved to png_path (if not
// empty). Returns when the window is closed (or Esc is pressed).
void plot_window(const std::string& title, int width, int height,
                 const std::function<void()>& draw, const std::string& png_path);

// Width of the panel's content area, for laying plots out side by side.
float plot_avail_w();
float plot_avail_h();
