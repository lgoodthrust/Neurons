// plotwin.cpp - result figures: a GLFW window with Dear ImGui + ImPlot, first frame saved as PNG.
#define _CRT_SECURE_NO_WARNINGS
#include "plotwin.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <GLFW/glfw3.h>               // includes the OpenGL 1.1 header (all we call directly)
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#ifdef _WIN32
#pragma comment(lib, "implot.lib")
#pragma comment(lib, "imgui.lib")
#pragma comment(lib, "opengl32.lib")
#endif

float plot_avail_w() { return ImGui::GetContentRegionAvail().x; }
float plot_avail_h() { return ImGui::GetContentRegionAvail().y; }

static void save_png(const std::string& path, int w, int h) {
    std::vector<unsigned char> px((size_t)w * h * 3), flipped(px.size());
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    for (int y = 0; y < h; ++y)                       // OpenGL rows run bottom-up
        std::copy(px.begin() + (size_t)(h - 1 - y) * w * 3, px.begin() + (size_t)(h - y) * w * 3,
                  flipped.begin() + (size_t)y * w * 3);
    if (stbi_write_png(path.c_str(), w, h, 3, flipped.data(), w * 3))
        std::printf("Plot saved: %s\n", path.c_str());
    else
        std::printf("Could not save %s\n", path.c_str());
}

void plot_window(const std::string& title, int width, int height,
                 const std::function<void()>& draw, const std::string& png_path) {
    if (!glfwInit()) { std::printf("(no window: GLFW init failed)\n"); return; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* win = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!win) { std::printf("(no window could be opened)\n"); glfwTerminate(); return; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImPlot::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    std::printf("(close the plot window, or press Esc, to continue)\n");

    int frame = 0;
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->Pos);
        ImGui::SetNextWindowSize(vp->Size);
        ImGui::Begin("##figure", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                          ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
        draw();
        ImGui::End();
        ImGui::Render();
        int fw, fh; glfwGetFramebufferSize(win, &fw, &fh);
        glViewport(0, 0, fw, fh);
        glClearColor(0.06f, 0.06f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        if (++frame == 4 && !png_path.empty()) save_png(png_path, fw, fh);   // after plots have auto-fitted
        glfwSwapBuffers(win);
        if (frame >= 4 && std::getenv("BRAIN_PLOT_AUTOCLOSE")) break;       // for automated tests
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
}
