// viz.cu - live visualizer ("brain in a vat"): window, camera, shaders, poke + task controls
#include "brain.cuh"
#ifdef BRAIN_VIZ
#ifdef _WIN32
#define NOMINMAX              // keep std::min / std::max usable
#define WIN32_LEAN_AND_MEAN
#include <windows.h>          // before GLEW, so APIENTRY is defined once (no C4005 warning)
#endif
#ifdef BRAIN_VIZ_STATIC
#define GLEW_STATIC           // GLEW compiled into the exe, not imported from glew32.dll
#endif
#include <GL/glew.h>          // must come before GLFW and cuda_gl_interop
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "audio.h"
#include <cctype>
#ifdef BRAIN_VIZ_UI
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#ifdef _WIN32
#pragma comment(lib, "imgui.lib")
#endif
#endif
#ifdef _WIN32
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "user32.lib")   // screen capture for the eyes (GDI)
#pragma comment(lib, "gdi32.lib")
#ifdef BRAIN_VIZ_STATIC
#pragma comment(lib, "user32.lib")   // a static GLFW needs these Windows system libraries
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "shell32.lib")
#endif
#endif
// =============================================================================
//  LIVE VISUALIZER — brain in a vat
// =============================================================================
//  Controls: left-drag rotate | wheel zoom | Up/Down = 2x faster/slower |
//            Space = pause | A = auto-rotate | R = reset view | Esc = quit
//            right-click (hold) = stimulate neurons under the cursor
//            1 = burst into group A | S = group A auto-stimulus on/off | G = highlight group A
//            [ ] = inhibition down/up | - = = background drive down/up | 0 = reset dials

// Glow of every neuron from the time since its last spike, written into the GL buffer.
__global__ void glow_kernel(const int* __restrict__ last_spike, float* __restrict__ glow,
                            int N, int now_step, float decay_steps) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float age = (float)(now_step - last_spike[n]);
    glow[n] = (age < 0.0f) ? 0.0f : __expf(-age / decay_steps);
}

// Activity graphs: how many neurons of each group spiked since step `since`
// (group 0 = eye/pixel neurons, 1 = other excitatory, 2 = inhibitory).
//   groups: 0 eye, 1 ear, 2 mouth, 3 other excitatory, 4 inhibitory;
//   out[5 + t] = spikes of mouth tone t's neurons.
//   Reads each neuron's exact spike count since the last call and resets it.
__global__ void group_count_kernel(unsigned* __restrict__ spike_n, int N, int eye_n, int ear_n,
                                   int mouth0, int N_E, int* __restrict__ out) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int g = n < eye_n ? 0 : n < eye_n + ear_n ? 1 : (n >= mouth0 && n < N_E) ? 2 : n < N_E ? 3 : 4;
    unsigned c = 0;
    if (n < N) { c = spike_n[n]; if (c) spike_n[n] = 0; }
    if (c && g == 2) atomicAdd(&out[5 + (n - mouth0) / MOUTH_PER], (int)c);
    // one atomic per warp per group instead of one per neuron
    for (int k = 0; k < 5; ++k) {
        const unsigned v = __reduce_add_sync(0xFFFFFFFFu, g == k ? c : 0u);
        if ((threadIdx.x & 31) == 0 && v) atomicAdd(&out[k], (int)v);
    }
}

// Live poking: these neurons receive stimulus input until step `until`.
__global__ void poke_kernel(int* poke_end, const int* __restrict__ idx, int count, int until) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count) poke_end[idx[k]] = until;
}

namespace viz {

struct Mat4 { float m[16]; };                      // column-major, like OpenGL

static Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
        for (int rr = 0; rr < 4; ++rr) {
            float v = 0;
            for (int k = 0; k < 4; ++k) v += a.m[k * 4 + rr] * b.m[c * 4 + k];
            r.m[c * 4 + rr] = v;
        }
    return r;
}
static Mat4 perspective(float fovy, float aspect, float zn, float zf) {
    const float f = 1.0f / std::tan(fovy / 2);
    Mat4 r{};
    r.m[0] = f / aspect; r.m[5] = f;
    r.m[10] = (zf + zn) / (zn - zf); r.m[11] = -1;
    r.m[14] = 2 * zf * zn / (zn - zf);
    return r;
}
static Mat4 look_at(float ex, float ey, float ez) {      // looking at the origin, y up
    float fx = -ex, fy = -ey, fz = -ez;
    float fl = std::sqrt(fx * fx + fy * fy + fz * fz); fx /= fl; fy /= fl; fz /= fl;
    float sx = fy * 0 - fz * 1, sy = fz * 0 - fx * 0, sz = fx * 1 - fy * 0;   // f x up
    float sl = std::sqrt(sx * sx + sy * sy + sz * sz); sx /= sl; sy /= sl; sz /= sl;
    const float ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;
    Mat4 r{};
    r.m[0] = sx; r.m[4] = sy; r.m[8]  = sz;
    r.m[1] = ux; r.m[5] = uy; r.m[9]  = uz;
    r.m[2] = -fx; r.m[6] = -fy; r.m[10] = -fz;
    r.m[12] = -(sx * ex + sy * ey + sz * ez);
    r.m[13] = -(ux * ex + uy * ey + uz * ez);
    r.m[14] = (fx * ex + fy * ey + fz * ez);
    r.m[15] = 1;
    return r;
}

static GLuint compile(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[2048]; glGetShaderInfoLog(sh, 2048, nullptr, log); std::fprintf(stderr, "shader: %s\n", log); }
    return sh;
}
static GLuint program(const char* vs, const char* fs) {
    GLuint p = glCreateProgram();
    GLuint a = compile(GL_VERTEX_SHADER, vs), b = compile(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, a); glAttachShader(p, b); glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[2048]; glGetProgramInfoLog(p, 2048, nullptr, log); std::fprintf(stderr, "link: %s\n", log); }
    glDeleteShader(a); glDeleteShader(b);
    return p;
}

// Neurons: glowing point sprites, additive blending
static const char* kPointVS = R"(#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in float type;     // 0 group A, 1 other E, 2 inhibitory
layout(location = 2) in float glow;
uniform mat4 mvp;
uniform float psize;
uniform float showA;                    // 1 = draw group A in green, 0 = same as other E cells
out vec3 col;
void main() {
    gl_Position = mvp * vec4(pos, 1.0);
    vec3 orange = vec3(1.0, 0.62, 0.22), blue = vec3(0.35, 0.65, 1.0);
    vec3 base;
    if (type > 2.5) {                       // sequence groups A, B, C, D
        int g = int(type - 2.5);
        vec3 pal = g == 0 ? vec3(0.35, 1.0, 0.45) : (g == 1 ? vec3(1.0, 0.92, 0.25)
                 : (g == 2 ? vec3(1.0, 0.35, 0.80) : vec3(0.30, 0.95, 1.0)));
        base = showA > 0.5 ? pal : orange;
    } else if (type < 0.5 && showA > 0.5) {
        base = vec3(0.35, 1.0, 0.45);
    } else {
        base = type < 1.5 ? orange : blue;
    }
    float rest = (type > 2.5 && showA > 0.5) ? 0.32 : 0.13;   // task groups stay visible at rest
    col = base * (rest + 1.5 * glow);
    gl_PointSize = psize * (0.7 + 1.8 * glow) / gl_Position.w;
}
)";
static const char* kPointFS = R"(#version 330 core
in vec3 col;
out vec4 frag;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    frag = vec4(col * exp(-3.0 * r2), 1.0);
}
)";
// Vat: thin lines
static const char* kLineVS = R"(#version 330 core
layout(location = 0) in vec3 pos;
uniform mat4 mvp;
void main() { gl_Position = mvp * vec4(pos, 1.0); }
)";
static const char* kLineFS = R"(#version 330 core
uniform vec3 color;
out vec4 frag;
void main() { frag = vec4(color, 1.0); }
)";

// Layout: a sphere, neurons spread evenly in a thick outer shell (like cortex on a ball).
static void make_layout(int N, int N_E, int N_A, std::vector<float>& xyzt, int seqK = 0, int seqS = 0) {
    std::mt19937 g(7);
    std::normal_distribution<float> G(0.f, 1.f);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    xyzt.resize((size_t)N * 4);
    const float R = 1.05f;                                     // sphere radius
    for (int n = 0; n < N; ++n) {
        const bool A = n < N_A;
        float dx, dy, dz;
        const int grp = (seqK > 0 && n < seqK * seqS) ? n / seqS : -1;
        for (;;) {                                             // random direction on the sphere
            dx = G(g); dy = G(g); dz = G(g);
            const float l = std::sqrt(dx * dx + dy * dy + dz * dz);
            dx /= l; dy /= l; dz /= l;
            if (grp < 0) break;
            // sequence groups: patches in a row across the front, A on the left
            const float th = (-48.f + 32.f * grp) * 3.14159265f / 180.f;
            const float cx = std::sin(th), cy = 0.10f, cz = std::cos(th);
            if (dx * cx + dy * cy + dz * cz > 0.965f * std::sqrt(cx * cx + cy * cy + cz * cz)) break;
        }
        const float shell = 0.80f + 0.20f * std::cbrt(U(g));   // shell thickness
        float* o = &xyzt[(size_t)n * 4];
        o[0] = dx * R * shell; o[1] = dy * R * shell; o[2] = dz * R * shell;
        o[3] = grp >= 0 ? 3.f + grp : (A ? 0.f : (n < N_E ? 1.f : 2.f));
    }
}

// Eyes: pixel neurons laid out as the picture itself, on the front of the sphere
// (pixel (x, y) -> longitude / latitude), so the screen shows up on the brain.
static void eye_layout(std::vector<float>& xyzt, int W, int H) {
    const float R = 1.05f, span_x = 1.9f, span_y = span_x * H / W;   // radians
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float lon = ((x + 0.5f) / W - 0.5f) * span_x;
            const float lat = (0.5f - (y + 0.5f) / H) * span_y;
            float* o = &xyzt[((size_t)y * W + x) * 4];
            o[0] = R * std::cos(lat) * std::sin(lon);
            o[1] = R * std::sin(lat);
            o[2] = R * std::cos(lat) * std::cos(lon);
            o[3] = 1.f;
        }
}

// Ears: one strip on each side of the head, low notes at the bottom, high at the top.
// Mouth: a strip under the eye picture, low tones on the left.
static void sphere_pt(float* o, float lon, float lat) {
    const float R = 1.05f;
    o[0] = R * std::cos(lat) * std::sin(lon); o[1] = R * std::sin(lat); o[2] = R * std::cos(lat) * std::cos(lon);
}
static void ear_mouth_layout(std::vector<float>& xyzt, int ear0, int ear_n, int mouth0, int mouth_n) {
    for (int i = 0; i < ear_n; ++i) {
        const int band = i / EAR_PER, k = i % EAR_PER, side = k & 1, j = k >> 1;   // alternate left / right
        const float lat = -0.75f + 1.5f * (band + 0.5f) / EAR_BANDS;
        const float lon = (side ? 1.0f : -1.0f) * (1.42f + 0.22f * ((j + 0.5f) / (EAR_PER / 2) - 0.5f));
        float* o = &xyzt[(size_t)(ear0 + i) * 4]; sphere_pt(o, lon, lat); o[3] = 1.f;
    }
    for (int i = 0; i < mouth_n; ++i) {
        const int tone = i / MOUTH_PER, k = i % MOUTH_PER;
        const float lon = 1.1f * ((tone + 0.5f) / MOUTH_TONES - 0.5f);
        const float lat = -0.66f - 0.14f * (k + 0.5f) / MOUTH_PER;
        float* o = &xyzt[(size_t)(mouth0 + i) * 4]; sphere_pt(o, lon, lat); o[3] = 1.f;
    }
}

#ifdef _WIN32
// Screen capture with GDI: StretchBlt shrinks the whole desktop to W x H in one call.
struct Eyes {
    int W = 0, H = 0, sw = 0, sh = 0;
    HDC screen = nullptr, mem = nullptr; HBITMAP bmp = nullptr, old = nullptr;
    unsigned char* px = nullptr;                               // BGRA, top row first
    bool init(int w, int h) {
        W = w; H = h;
        SetProcessDPIAware();                                  // real pixels, not scaled ones
        sw = GetSystemMetrics(SM_CXSCREEN); sh = GetSystemMetrics(SM_CYSCREEN);
        screen = GetDC(nullptr); mem = CreateCompatibleDC(screen);
        BITMAPINFO bi{}; bi.bmiHeader.biSize = sizeof(bi.bmiHeader);
        bi.bmiHeader.biWidth = W; bi.bmiHeader.biHeight = -H;  // negative: top-down
        bi.bmiHeader.biPlanes = 1; bi.bmiHeader.biBitCount = 32; bi.bmiHeader.biCompression = BI_RGB;
        bmp = CreateDIBSection(mem, &bi, DIB_RGB_COLORS, (void**)&px, nullptr, 0);
        if (!bmp) return false;
        old = (HBITMAP)SelectObject(mem, bmp);
        SetStretchBltMode(mem, HALFTONE); SetBrushOrgEx(mem, 0, 0, nullptr);
        return true;
    }
    void grab(std::vector<float>& lum) {                       // brightness 0..1 per pixel
        StretchBlt(mem, 0, 0, W, H, screen, 0, 0, sw, sh, SRCCOPY);
        GdiFlush();
        for (int i = 0; i < W * H; ++i) {
            const unsigned char* c = px + 4 * i;
            lum[i] = (0.0722f * c[0] + 0.7152f * c[1] + 0.2126f * c[2]) / 255.f;
        }
    }
    ~Eyes() {
        if (mem) { SelectObject(mem, old); DeleteObject(bmp); DeleteDC(mem); }
        if (screen) ReleaseDC(nullptr, screen);
    }
};
#endif

static void make_vat(std::vector<float>& v) {                  // cylinder wireframe
    const float R = 1.55f, y0 = -1.25f, y1 = 1.35f;
    const int seg = 72;
    auto ring = [&](float y) {
        for (int i = 0; i < seg; ++i) {
            const float a0 = 6.2831853f * i / seg, a1 = 6.2831853f * (i + 1) / seg;
            v.insert(v.end(), { R * std::cos(a0), y, R * std::sin(a0), R * std::cos(a1), y, R * std::sin(a1) });
        }
    };
    ring(y0); ring(y1); ring(y0 + 0.05f);
    for (int i = 0; i < 12; ++i) {
        const float a = 6.2831853f * i / 12;
        v.insert(v.end(), { R * std::cos(a), y0, R * std::sin(a), R * std::cos(a), y1, R * std::sin(a) });
    }
}

static bool ui_wants_mouse() {
#ifdef BRAIN_VIZ_UI
    return ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse;
#else
    return false;
#endif
}
static bool ui_wants_keys() {
#ifdef BRAIN_VIZ_UI
    return ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureKeyboard;
#else
    return false;
#endif
}

struct Cam { float yaw = 0.0f, pitch = 0.30f, dist = 4.2f; bool drag = false; double lx = 0, ly = 0; };
static Cam* g_cam = nullptr;

}  // namespace viz

int run_live(DevState& s, Params& p, int blocks, int threads, bool seq_live, int senses,
             const LiveCtx& ctx) {
    const std::vector<float>* layout = ctx.layout;
    const bool eyes_on = (senses & SENSE_EYES) != 0;
    using namespace viz;
    const int N = p.N;
    if (!glfwInit()) { std::fprintf(stderr, "GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* win = glfwCreateWindow(1400, 900, "Brain in a vat", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "Could not open a window\n"); glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::fprintf(stderr, "GLEW init failed\n"); return 1; }

    Cam cam; g_cam = &cam;
    glfwSetMouseButtonCallback(win, [](GLFWwindow* w, int b, int act, int) {
        if (b != GLFW_MOUSE_BUTTON_LEFT) return;
        if (act == GLFW_PRESS && ui_wants_mouse()) return;           // click is for the panel
        g_cam->drag = act == GLFW_PRESS; glfwGetCursorPos(w, &g_cam->lx, &g_cam->ly);
    });
    glfwSetCursorPosCallback(win, [](GLFWwindow*, double x, double y) {
        if (!g_cam->drag) return;
        g_cam->yaw   -= (float)(x - g_cam->lx) * 0.006f;   // drag right -> scene turns right
        g_cam->pitch += (float)(y - g_cam->ly) * 0.006f;
        g_cam->pitch  = std::max(-1.45f, std::min(1.45f, g_cam->pitch));
        g_cam->lx = x; g_cam->ly = y;
    });
    glfwSetScrollCallback(win, [](GLFWwindow*, double, double dy) {
        if (ui_wants_mouse()) return;
        g_cam->dist = std::max(1.2f, std::min(12.f, g_cam->dist * (dy > 0 ? 0.9f : 1.1f)));
    });

#ifdef BRAIN_VIZ_UI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;                 // don't write imgui.ini next to the exe
    ImGui::StyleColorsDark();
    ImGui::GetStyle().Alpha = 0.92f;
    ImGui_ImplGlfw_InitForOpenGL(win, true);              // chains to the callbacks above
    ImGui_ImplOpenGL3_Init("#version 330");
#endif

    // ---- geometry ----
    std::vector<float> xyzt;
    make_layout(N, p.N_E, p.N_A, xyzt, seq_live ? p.seq_K : 0, p.seq_size);
    const int EW = EYE_W, EH = EYE_H;
    const int EN = eyes_on ? EW * EH : 0;                              // eye (pixel) neurons: 0 .. EN-1
    const int AN = (senses & SENSE_EARS) ? EAR_BANDS * EAR_PER : 0;    // ear neurons: EN .. EN+AN-1
    const int MN = (senses & SENSE_MOUTH) ? MOUTH_TONES * MOUTH_PER : 0;
    const int mouth0 = p.N_E - MN;                                     // mouth neurons: the last E cells
    if (layout && layout->size() == (size_t)N * 4) {
        xyzt = *layout;                        // anatomy: positions come from the network builder
    } else {
        if (eyes_on) eye_layout(xyzt, EW, EH);
        ear_mouth_layout(xyzt, EN, AN, mouth0, MN);
    }
    std::vector<float> vat;  make_vat(vat);
    GLuint vao, vbo_pos, vbo_glow, vao_vat, vbo_vat;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo_pos); glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
    glBufferData(GL_ARRAY_BUFFER, xyzt.size() * sizeof(float), xyzt.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(0); glEnableVertexAttribArray(1);
    glGenBuffers(1, &vbo_glow); glBindBuffer(GL_ARRAY_BUFFER, vbo_glow);
    glBufferData(GL_ARRAY_BUFFER, (size_t)N * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);
    glGenVertexArrays(1, &vao_vat); glBindVertexArray(vao_vat);
    glGenBuffers(1, &vbo_vat); glBindBuffer(GL_ARRAY_BUFFER, vbo_vat);
    glBufferData(GL_ARRAY_BUFFER, vat.size() * sizeof(float), vat.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    cudaGraphicsResource* glow_res = nullptr;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&glow_res, vbo_glow, cudaGraphicsMapFlagsWriteDiscard));

    GLuint prog_pts = program(kPointVS, kPointFS), prog_vat = program(kLineVS, kLineFS);
    const GLint u_mvp = glGetUniformLocation(prog_pts, "mvp"), u_ps = glGetUniformLocation(prog_pts, "psize");
    const GLint u_showA = glGetUniformLocation(prog_pts, "showA");
    const GLint u_mvp2 = glGetUniformLocation(prog_vat, "mvp"), u_col = glGetUniformLocation(prog_vat, "color");
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_MULTISAMPLE);

    // ---- poking: device index buffer ----
    int* d_poke_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_poke_idx, (size_t)N * sizeof(int)));
    std::vector<int> poke_list; poke_list.reserve(N);
    auto poke = [&](const std::vector<int>& ids, int until) {
        if (ids.empty()) return;
        CUDA_CHECK(cudaMemcpy(d_poke_idx, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        poke_kernel<<<((int)ids.size() + 255) / 256, 256>>>(s.poke_end, d_poke_idx, (int)ids.size(), until);
    };
    std::vector<int> groupA_ids(p.N_A);
    for (int i = 0; i < p.N_A; ++i) groupA_ids[i] = i;

    // dials (drive scales the background Poisson rates; inhibition scales all inhibitory input)
    const float base_lamE_E = p.lamE_branch_E, base_lamE_I = p.lamE_branch_I;
    double inh = 1.0, drive = 1.0;
    auto apply_dials = [&]() {
        p.inh_scale     = (float)inh;
        p.lamE_branch_E = base_lamE_E * (float)drive;
        p.lamE_branch_I = base_lamE_I * (float)drive;
        p.expE_E = std::exp(-p.lamE_branch_E);
        p.expE_I = std::exp(-p.lamE_branch_I);
    };
    bool kOne = false, kS = false, kLb = false, kRb = false, kMi = false, kEq = false, k0 = false, kG = false;
    bool showA = seq_live || layout != nullptr;   // G: colour task groups / regions / group A
    bool kT = false, kComma = false, kDot = false, k2 = false, k3 = false, k4 = false;
    const float base_Ap = p.A_plus, base_Am = p.A_minus;
    double lr = 1.0;
    std::vector<std::vector<int>> seq_ids(p.seq_K);
    for (int g2 = 0; g2 < p.seq_K; ++g2)
        for (int i = 0; i < p.seq_size; ++i) seq_ids[g2].push_back(g2 * p.seq_size + i);
    p.auto_stim = 0;                    // live view starts quiet: you do the stimulating (S turns it on)

    double disp_speed = 0.0, disp_fps = 0.0;  // readouts for the panel

    // ---- eyes: capture the screen ~30 times a second, upload brightness to the pixel neurons ----
    bool eyes_live = false; double eye_gain = 1.0, t_eye = 0.0;
    const float base_eye_gain = p.eye_gain;
    std::vector<float> lum;
    GLuint eye_tex = 0;
#ifdef _WIN32
    Eyes cap;
    if (eyes_on) {
        if (cap.init(EW, EH)) {
            eyes_live = true; lum.assign((size_t)EW * EH, 0.f);
            glGenTextures(1, &eye_tex); glBindTexture(GL_TEXTURE_2D, eye_tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, EW, EH, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
            std::printf("Eyes: capturing %dx%d screen -> %dx%d pixels -> neurons 0..%d\n",
                        cap.sw, cap.sh, EW, EH, EW * EH - 1);
        } else std::fprintf(stderr, "Eyes: screen capture failed to start\n");
    }
#endif
    bool eyes_open = eyes_live;           // panel checkbox: eyes open / closed

    // ---- ears and mouth ----
    const float F_EAR0 = 20.f, F_EAR1 = 16000.f;     // ear bands: 20 Hz .. 16 kHz
    const float F_MOUTH0 = 20.f, F_MOUTH1 = 16000.f; // mouth tones: 20 Hz .. 16 kHz
    bool ears_live = AN > 0 && ears_start(), ears_open = ears_live;
    bool mouth_live = MN > 0 && mouth_start(), mouth_open = false;   // mouth starts muted: you unmute it
    if (AN > 0) std::printf(ears_live ? "Ears: listening to the default output device\n"
                                      : "Ears: could not open the output device for listening\n");
    if (MN > 0) std::printf(mouth_live ? "Mouth: ready (muted until you tick 'Mouth on')\n"
                                       : "Mouth: could not open sound output\n");
    double ear_gain = 1.0, mouth_vol = 0.3;
    std::vector<float> bands(EAR_BANDS, 0.f), mouth_amp(MOUTH_TONES, 0.f);
    std::vector<float> m_fast(MOUTH_TONES, 0.f), m_slow(MOUTH_TONES, 0.f);   // smoothed tone rates (Hz)
    std::vector<float> sense((size_t)p.eye_n, 0.f);
    bool eye_tick = false;                // a new screen frame arrived this frame
    float scr_bright = 0.f, scr_change = 0.f;
    std::vector<float> prev_lum(lum.size(), 0.f);

    // ---- activity graphs ----
    const int HIST = 600;                            // samples kept (one per 20 ms simulated = 12 s)
    std::vector<float> h_eye(HIST, 0.f), h_rest(HIST, 0.f), h_inh(HIST, 0.f), h_bright(HIST, 0.f), h_chg(HIST, 0.f);
    std::vector<float> h_ear(HIST, 0.f), h_mouth(HIST, 0.f), h_loud(HIST, 0.f), h_da(HIST, 0.f);
    int hpos = 0, prev_step = 0;
    long long acc_cnt[5] = { 0, 0, 0, 0, 0 }; double acc_s = 0.0;
    float cur_rate[5] = { 0.f, 0.f, 0.f, 0.f, 0.f }, cur_loud = 0.f;
    const int NCNT = 5 + MOUTH_TONES;
    int* d_cnt = nullptr; CUDA_CHECK(cudaMalloc(&d_cnt, NCNT * sizeof(int)));
    // what the pixel neurons are doing, as a picture (same 192x108 grid as the screen)
    GLuint act_tex = 0;
    std::vector<float> act(EN); std::vector<unsigned char> act_rgba((size_t)EN * 4);
    if (EN > 0) {
        glGenTextures(1, &act_tex); glBindTexture(GL_TEXTURE_2D, act_tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, EW, EH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    }

    // ---- run loop ----
    double speed = 1.0;                 // simulated seconds per real second
    char seq_buf[96];
    auto seq_status = [&]() -> const char* {
        if (!seq_live) std::snprintf(seq_buf, sizeof(seq_buf), "auto-stim %s", p.auto_stim ? "on" : "off");
        else std::snprintf(seq_buf, sizeof(seq_buf), "SEQUENCE training %s | learning x%.2f",
                           p.train1 > 0 ? "ON" : "off", lr);
        return seq_buf;
    };
    bool show_graphs = true;

    bool paused = false, autorot = false;  // A toggles slow auto-rotation
    bool kUp = false, kDn = false, kSp = false, kA = false;
    int step = 0;

    // ---- saving the brain (brains/<name>.brain next to the exe) ----
    char brain_name[64];
    std::snprintf(brain_name, sizeof(brain_name), "%s", ctx.name.c_str());
    std::string save_msg;
    auto save_now = [&]() {
        std::string name(brain_name);
        for (char& c : name)                              // keep the file name safe
            if (!(std::isalnum((unsigned char)c) || c == '-' || c == '_')) c = '_';
        if (name.empty()) name = "mybrain";
        const double age = ctx.age_s + step * p.dt / 1000.0;
        save_brain("brains/" + name + ".brain", s, p, ctx.nnz, senses, layout, age, save_msg);
    };

    // ---- reward learning (dopamine) ----
    const bool reward_ok = s.elig != nullptr;
    double reward_gain = 1.0, elig_tau_s = 1.0, dopamine = 0.0;   // dopamine: shown in the graphs only
    double reward_s = 0.0, punish_s = 0.0;           // seconds of reward / punishment given
    bool btn_reward = false, btn_punish = false;     // panel buttons held down (from last frame)
    bool mouth_level = true;                         // mouth follows firing level (else: changes only)
    // While held, dopamine flows continuously: each frame applies the recent eligibility, scaled by
    // the simulated time that frame covered (strength = change per 0.1 s of holding).
    auto apply_dopamine = [&](float sign, double sim_s) {
        if (!reward_ok || sim_s <= 0.0) return;
        reward_kernel<<<(p.N_E + 127) / 128, 128>>>(s.in_ptr, s.in_eend, s.in_w, s.elig, s.elig_t, p.N_E, step,
                                                     p.k_elig, sign * (float)(reward_gain * sim_s / 0.1), p.w_max);
        CUDA_CHECK(cudaGetLastError());
        dopamine = sign;
        (sign > 0 ? reward_s : punish_s) += sim_s;
    };
    auto set_training = [&](bool on) {
        if (!on) p.train1 = 0;
        else { p.train0 = step - step % p.seq_cycle + p.seq_cycle; p.train1 = 2000000000; }  // from next cycle
    };
    double carry = 0.0;                 // fractional epochs owed
    const int burst10 = (int)std::lround(10.0 / p.dt);        // 10 ms of stimulation, whatever the time step
    const double epoch_s = p.M * p.dt / 1000.0;
    double t_last = glfwGetTime(), t_title = t_last, sim_acc = 0.0;
    int frames = 0;
    std::printf("Live view: %d neurons.\n"
                "  Left-drag rotate | wheel zoom | Up/Down speed | Space pause | A auto-rotate | R reset view | Esc quit\n"
                "  Right-click (hold) poke | 1 burst group A | S group-A auto-stimulus on/off | G highlight group A\n"
                "  [ ] inhibition down/up | - = background drive down/up | 0 reset dials\n%s", N,
                seq_live ? "  SEQUENCE: T training on/off | 1-4 cue group A-D | , . learning rate down/up\n" : "");

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        auto edge = [&](int key, bool& prev) {
            const bool now = glfwGetKey(win, key) == GLFW_PRESS && !ui_wants_keys();
            const bool hit = now && !prev; prev = now; return hit;
        };
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(win, 1);
        if (edge(GLFW_KEY_UP, kUp))   speed = std::min(speed * 2.0, 64.0);
        if (edge(GLFW_KEY_DOWN, kDn)) speed = std::max(speed / 2.0, 1.0 / 64.0);
        if (edge(GLFW_KEY_SPACE, kSp)) paused = !paused;
        if (edge(GLFW_KEY_A, kA))      autorot = !autorot;
        if (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS) { cam = Cam(); }
        if (edge(GLFW_KEY_S, kS)) p.auto_stim = p.auto_stim ? 0 : 1;
        if (edge(GLFW_KEY_G, kG)) showA = !showA;
        // hold Enter (or the Reward button) = reward, hold Backspace (or Punish) = punish
        const bool hold_reward = reward_ok && ((glfwGetKey(win, GLFW_KEY_ENTER) == GLFW_PRESS && !ui_wants_keys()) || btn_reward);
        const bool hold_punish = reward_ok && ((glfwGetKey(win, GLFW_KEY_BACKSPACE) == GLFW_PRESS && !ui_wants_keys()) || btn_punish);
        if (edge(GLFW_KEY_LEFT_BRACKET, kLb))  { inh = std::max(0.1, inh / 1.1);  apply_dials(); }
        if (edge(GLFW_KEY_RIGHT_BRACKET, kRb)) { inh = std::min(3.0, inh * 1.1);  apply_dials(); }
        if (edge(GLFW_KEY_MINUS, kMi)) { drive = std::max(0.5, drive / 1.02); apply_dials(); }
        if (edge(GLFW_KEY_EQUAL, kEq)) { drive = std::min(1.6, drive * 1.02); apply_dials(); }
        if (edge(GLFW_KEY_0, k0))      { inh = 1.0; drive = 1.0; apply_dials(); }
        if (!seq_live) {
            if (edge(GLFW_KEY_1, kOne)) poke(groupA_ids, step + burst10);         // 10 ms burst
        } else {
            const int len = p.seq_len;
            if (edge(GLFW_KEY_1, kOne)) poke(seq_ids[0], step + len);        // cue a task group
            if (edge(GLFW_KEY_2, k2))   poke(seq_ids[1], step + len);
            if (edge(GLFW_KEY_3, k3))   poke(seq_ids[2], step + len);
            if (edge(GLFW_KEY_4, k4))   poke(seq_ids[3], step + len);
            if (edge(GLFW_KEY_T, kT)) set_training(!(p.train1 > step));      // training on/off
            if (edge(GLFW_KEY_COMMA, kComma)) lr = std::max(0.25, lr / 1.25);
            if (edge(GLFW_KEY_PERIOD, kDot))  lr = std::min(32.0, lr * 1.25);
            p.A_plus = base_Ap * (float)lr; p.A_minus = base_Am * (float)lr;
        }

        const double now = glfwGetTime();
        const double dt_wall = std::min(now - t_last, 0.1);
        t_last = now;

#ifdef _WIN32
        if (eyes_live && now - t_eye > 1.0 / 30.0) {
            t_eye = now;
            if (eyes_open) cap.grab(lum); else std::fill(lum.begin(), lum.end(), 0.f);
            double sb = 0.0, sc = 0.0;
            for (size_t i = 0; i < lum.size(); ++i) { sb += lum[i]; sc += std::fabs(lum[i] - prev_lum[i]); }
            scr_bright = (float)(sb / lum.size()); scr_change = (float)(sc / lum.size());
            prev_lum = lum; eye_tick = true;
            glBindTexture(GL_TEXTURE_2D, eye_tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, EW, EH, GL_BGRA, GL_UNSIGNED_BYTE, cap.px);
        }
#endif
        p.eye_gain = base_eye_gain;
        if (p.eye_n > 0) {                           // sensory input: pixel brightness, then ear bands
            for (int i = 0; i < EN; ++i) sense[i] = eyes_live ? lum[i] * (float)eye_gain : 0.f;
            if (AN > 0) {
                if (ears_live && ears_open) ears_bands(bands.data(), EAR_BANDS, F_EAR0, F_EAR1);
                else std::fill(bands.begin(), bands.end(), 0.f);
                for (int i = 0; i < AN; ++i) sense[EN + i] = bands[i / EAR_PER] * (float)ear_gain;
            }
            CUDA_CHECK(cudaMemcpy(s.eye, sense.data(), sense.size() * sizeof(float), cudaMemcpyHostToDevice));
        }

        // advance the simulation by (speed x wall time), in whole epochs, capped per frame
        if (!paused) {
            carry += speed * dt_wall / epoch_s;
            int epochs = (int)carry;
            const int cap = 400;                       // ~40 ms of simulated time per frame at most
            if (epochs > cap) { epochs = cap; carry = 0.0; } else carry -= epochs;
            if (epochs > 0) {
                int begin = step, end = step + epochs * p.M;
                void* args[] = { &s, &p, &begin, &end };
                CUDA_CHECK(cudaLaunchCooperativeKernel((void*)simulate, dim3(blocks), dim3(threads), args, 0, 0));
                step = end;
                sim_acc += epochs * epoch_s;
                if (hold_reward && !hold_punish) apply_dopamine(+1.0f, epochs * epoch_s);
                if (hold_punish && !hold_reward) apply_dopamine(-1.0f, epochs * epoch_s);
            }
        }

        // glow: fade over ~0.15 s of real time, whatever the speed
        float* d_glow = nullptr; size_t bytes = 0;
        CUDA_CHECK(cudaGraphicsMapResources(1, &glow_res, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_glow, &bytes, glow_res));
        const float decay_steps = (float)std::max(2.0, 0.15 * speed * 1000.0 / p.dt);
        glow_kernel<<<(N + 255) / 256, 256>>>(s.last_spike, d_glow, N, step, decay_steps);
        if (eye_tick && EN > 0) {                    // pixel neurons' glow -> "what its eye neurons do" image
            CUDA_CHECK(cudaMemcpy(act.data(), d_glow, (size_t)EN * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < EN; ++i) {
                const float v = std::min(1.f, act[i] * 1.5f);
                act_rgba[4 * i + 0] = (unsigned char)(255 * std::min(1.f, v * 1.6f));          // black -> orange -> white
                act_rgba[4 * i + 1] = (unsigned char)(255 * std::max(0.f, std::min(1.f, v * 1.6f - 0.45f)));
                act_rgba[4 * i + 2] = (unsigned char)(255 * std::max(0.f, v * 2.f - 1.f));
                act_rgba[4 * i + 3] = 255;
            }
            glBindTexture(GL_TEXTURE_2D, act_tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, EW, EH, GL_RGBA, GL_UNSIGNED_BYTE, act_rgba.data());
        }
        eye_tick = false;
        if (step > prev_step) {                      // firing rates of each group since the last frame
            CUDA_CHECK(cudaMemset(d_cnt, 0, NCNT * sizeof(int)));
            group_count_kernel<<<(N + 255) / 256, 256>>>(s.spike_n, N, EN, AN, MN > 0 ? mouth0 : p.N_E,
                                                         p.N_E, d_cnt);
            int c[5 + MOUTH_TONES]; CUDA_CHECK(cudaMemcpy(c, d_cnt, sizeof(c), cudaMemcpyDeviceToHost));
            for (int k = 0; k < 5; ++k) acc_cnt[k] += c[k];
            const double fr_s = (step - prev_step) * p.dt / 1000.0;
            acc_s += fr_s;
            prev_step = step;
            if (MN > 0) {                            // mouth: tone loudness = how far its neurons fire above their usual rate
                const float af = (float)std::min(1.0, fr_s / 0.05), as = (float)std::min(1.0, fr_s / 5.0);
                for (int t = 0; t < MOUTH_TONES; ++t) {
                    const float r = (float)(c[5 + t] / (MOUTH_PER * fr_s));
                    m_fast[t] += af * (r - m_fast[t]);                // ~50 ms average
                    m_slow[t] += as * (m_fast[t] - m_slow[t]);        // ~5 s average = "usual"
                    const float drive = mouth_level ? m_fast[t] - 2.0f          // level: above ~background
                                                    : m_fast[t] - m_slow[t];   // changes: above its own usual
                    mouth_amp[t] = std::min(1.f, std::max(0.f, drive / 10.f));
                }
            }
            dopamine *= std::exp(-fr_s / 0.2);            // dopamine burst fades over ~0.2 s
            if (acc_s >= 0.02) {                     // one graph sample per 20 ms of simulated time
                const double sz[5] = { (double)std::max(EN, 1), (double)std::max(AN, 1), (double)std::max(MN, 1),
                                       (double)std::max(p.N_E - EN - AN - MN, 1), (double)(N - p.N_E) };
                for (int k = 0; k < 5; ++k) { cur_rate[k] = (float)(acc_cnt[k] / sz[k] / acc_s); acc_cnt[k] = 0; }
                acc_s = 0.0;
                float ld = 0.f; for (float b : bands) ld += b; cur_loud = 100.f * ld / EAR_BANDS;
                h_eye[hpos] = cur_rate[0]; h_ear[hpos] = cur_rate[1]; h_mouth[hpos] = cur_rate[2];
                h_rest[hpos] = cur_rate[3]; h_inh[hpos] = cur_rate[4]; h_loud[hpos] = cur_loud; h_da[hpos] = (float)dopamine;
                h_bright[hpos] = scr_bright * 100.f; h_chg[hpos] = scr_change * 100.f;
                hpos = (hpos + 1) % HIST;
            }
        }
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &glow_res, 0));

        if (mouth_live) mouth_update(mouth_amp.data(), MOUTH_TONES, F_MOUTH0, F_MOUTH1,
                                     (mouth_open && !paused) ? (float)mouth_vol : 0.f);

        // ---- draw ----
        if (autorot && !cam.drag) cam.yaw += (float)dt_wall * 0.12f;
        int fw, fh; glfwGetFramebufferSize(win, &fw, &fh);
        glViewport(0, 0, fw, fh);
        glClearColor(0.01f, 0.012f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        const float ex = cam.dist * std::cos(cam.pitch) * std::sin(cam.yaw);
        const float ey = cam.dist * std::sin(cam.pitch);
        const float ez = cam.dist * std::cos(cam.pitch) * std::cos(cam.yaw);
        const Mat4 mvp = mul(perspective(0.8f, (float)fw / std::max(fh, 1), 0.05f, 50.f), look_at(ex, ey, ez));

        // right-click: stimulate the neurons under the cursor (ray vs. the sphere)
        if (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS && !ui_wants_mouse()) {
            double mx, my; int ww, wh;
            glfwGetCursorPos(win, &mx, &my); glfwGetWindowSize(win, &ww, &wh);
            const float nx = (float)(2.0 * mx / std::max(ww, 1) - 1.0), ny = (float)(1.0 - 2.0 * my / std::max(wh, 1));
            float fx = -ex, fy = -ey, fz = -ez;                         // camera basis (as in look_at)
            const float fl = std::sqrt(fx * fx + fy * fy + fz * fz); fx /= fl; fy /= fl; fz /= fl;
            float sx = -fz, sy = 0.0f, sz = fx;                          // f x (0,1,0)
            const float sl = std::sqrt(sx * sx + sz * sz); sx /= sl; sz /= sl;
            const float ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;
            const float th = std::tan(0.4f), asp = (float)fw / std::max(fh, 1);
            float dx = fx + nx * th * asp * sx + ny * th * ux;
            float dy = fy + nx * th * asp * sy + ny * th * uy;
            float dz = fz + nx * th * asp * sz + ny * th * uz;
            const float dl = std::sqrt(dx * dx + dy * dy + dz * dz); dx /= dl; dy /= dl; dz /= dl;
            const float R = 1.05f;                                       // sphere radius used by the layout
            const float b = ex * dx + ey * dy + ez * dz, cc = ex * ex + ey * ey + ez * ez - R * R;
            const float disc = b * b - cc;
            if (disc >= 0.0f) {
                const float tq = -b - std::sqrt(disc);                   // nearest hit
                float hx = ex + tq * dx, hy = ey + tq * dy, hz = ez + tq * dz;
                const float hl = std::sqrt(hx * hx + hy * hy + hz * hz); hx /= hl; hy /= hl; hz /= hl;
                poke_list.clear();
                for (int n = 0; n < N; ++n) {
                    const float* o = &xyzt[(size_t)n * 4];
                    const float ol = std::sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]);
                    if ((o[0] * hx + o[1] * hy + o[2] * hz) > 0.97f * ol) poke_list.push_back(n);  // ~14 deg
                }
                poke(poke_list, step + burst10);                             // keep them stimulated while held
            }
        }

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(prog_vat);
        glUniformMatrix4fv(u_mvp2, 1, GL_FALSE, mvp.m);
        glUniform3f(u_col, 0.10f, 0.16f, 0.22f);
        glBindVertexArray(vao_vat);
        glDrawArrays(GL_LINES, 0, (GLsizei)(vat.size() / 3));

        glBlendFunc(GL_ONE, GL_ONE);                      // additive: overlapping glows add up
        glUseProgram(prog_pts);
        glUniformMatrix4fv(u_mvp, 1, GL_FALSE, mvp.m);
        const float psize = 9.0f * (float)fh / 900.0f * std::sqrt(30000.0f / std::max(N, 1000));
        glUniform1f(u_ps, psize);
        glUniform1f(u_showA, showA ? 1.0f : 0.0f);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, N);

#ifdef BRAIN_VIZ_UI
        // ---- control panel ----
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(350, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Brain controls");
        ImGui::Text("%d neurons   t = %.1f s", N, step * p.dt / 1000.0);
        ImGui::Text("%.2fx real time   %.0f fps%s", disp_speed, disp_fps, paused ? "   PAUSED" : "");

        ImGui::SeparatorText("Simulation");
        float sp_f = (float)speed;
        if (ImGui::SliderFloat("Speed", &sp_f, 1.0f / 64, 64.0f, "%.3gx", ImGuiSliderFlags_Logarithmic)) speed = sp_f;
        ImGui::Checkbox("Pause", &paused);

        ImGui::SeparatorText("Network");
        float inh_f = (float)inh, drv_f = (float)drive;
        if (ImGui::SliderFloat("Inhibition", &inh_f, 0.1f, 3.0f, "%.2fx")) { inh = inh_f; apply_dials(); }
        if (ImGui::SliderFloat("Drive", &drv_f, 0.5f, 1.6f, "%.3fx")) { drive = drv_f; apply_dials(); }
        if (ImGui::Button("Reset inhibition + drive")) { inh = 1.0; drive = 1.0; apply_dials(); }
        if (!seq_live) {
            bool auto_on = p.auto_stim != 0;
            if (ImGui::Checkbox("Group A auto-stimulus (every 200 ms)", &auto_on)) p.auto_stim = auto_on ? 1 : 0;
            if (ImGui::Button("Burst group A")) poke(groupA_ids, step + burst10);
        } else {
            ImGui::SeparatorText("Sequence task");
            bool tr_on = p.train1 > step;
            if (ImGui::Checkbox("Training  A -> B -> C -> D", &tr_on)) set_training(tr_on);
            float lr_f = (float)lr;
            if (ImGui::SliderFloat("Learning rate", &lr_f, 0.25f, 32.0f, "%.2fx", ImGuiSliderFlags_Logarithmic)) lr = lr_f;
            p.A_plus = base_Ap * (float)lr; p.A_minus = base_Am * (float)lr;
            ImGui::Text("Cue a group:");
            const char* cue_names[4] = { " A ", " B ", " C ", " D " };
            for (int g2 = 0; g2 < p.seq_K && g2 < 4; ++g2) {
                ImGui::SameLine();
                if (ImGui::Button(cue_names[g2])) poke(seq_ids[g2], step + p.seq_len);
            }
        }

        if (eyes_live) {
            ImGui::SeparatorText("Eyes");
            ImGui::Checkbox("Eyes open", &eyes_open);
            float eg = (float)eye_gain;
            if (ImGui::SliderFloat("Eye strength", &eg, 0.0f, 4.0f, "%.2fx")) eye_gain = eg;
            ImGui::Image((ImTextureID)(intptr_t)eye_tex, ImVec2(EW * 1.7f, EH * 1.7f));
            ImGui::TextDisabled("What it sees: 192x108, one neuron per pixel");
        }
        if (reward_ok) {
            ImGui::SeparatorText("Reward learning (dopamine)");
            bool da = p.da_mode != 0;
            if (ImGui::Checkbox("Learn only from rewards", &da)) p.da_mode = da ? 1 : 0;
            if (!da) ImGui::TextDisabled("(off: plain STDP, learns from everything)");
            float rs = (float)reward_gain;
            if (ImGui::SliderFloat("Reward strength", &rs, 0.1f, 10.0f, "%.1f", ImGuiSliderFlags_Logarithmic)) reward_gain = rs;
            float mw = (float)elig_tau_s;
            if (ImGui::SliderFloat("Memory window", &mw, 0.2f, 5.0f, "%.1f s", ImGuiSliderFlags_Logarithmic)) {
                elig_tau_s = mw; p.k_elig = p.dt / (float)(elig_tau_s * 1000.0);
            }
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.55f, 0.25f, 1));
            ImGui::Button("  Reward (hold Enter)  ");
            btn_reward = ImGui::IsItemActive();
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.2f, 0.2f, 1));
            ImGui::Button("  Punish (hold Backspace)  ");
            btn_punish = ImGui::IsItemActive();
            ImGui::PopStyleColor();
            ImGui::Text("Reward given %.1f s   punishment %.1f s", reward_s, punish_s);
            ImGui::TextDisabled("Strengthens what was active in the last ~%.1f s", elig_tau_s);
        }
        if (AN > 0) {
            ImGui::SeparatorText("Ears");
            if (!ears_live) ImGui::TextDisabled("(no output device to listen to)");
            ImGui::Checkbox("Ears on", &ears_open);
            float eg2 = (float)ear_gain;
            if (ImGui::SliderFloat("Ear strength", &eg2, 0.0f, 4.0f, "%.2fx")) ear_gain = eg2;
            ImGui::PlotHistogram("##ears", bands.data(), EAR_BANDS, 0, "what it hears (20 Hz - 16 kHz)",
                                 0.f, 1.f, ImVec2(330, 55));
        }
        if (MN > 0) {
            ImGui::SeparatorText("Mouth");
            if (!mouth_live) ImGui::TextDisabled("(no sound output)");
            ImGui::Checkbox("Mouth on (plays through your speakers)", &mouth_open);
            float mv = (float)mouth_vol;
            if (ImGui::SliderFloat("Volume", &mv, 0.0f, 1.0f, "%.2f")) mouth_vol = mv;
            ImGui::PlotHistogram("##mouth", mouth_amp.data(), MOUTH_TONES, 0, "what it says (20 Hz - 16 kHz)",
                                 0.f, 1.f, ImVec2(330, 55));
            ImGui::Checkbox("Mouth follows level (off: only changes)", &mouth_level);
            if (AN > 0) ImGui::TextDisabled("Its ears hear its own mouth: keep the volume low");
        }

        ImGui::SeparatorText("Brain file");
        ImGui::Text("Age: %.1f min of experience", (ctx.age_s + step * p.dt / 1000.0) / 60.0);
        ImGui::SetNextItemWidth(170);
        ImGui::InputText("##brainname", brain_name, sizeof(brain_name));
        ImGui::SameLine();
        if (ImGui::Button("Save brain")) save_now();
        if (!save_msg.empty()) ImGui::TextWrapped("%s", save_msg.c_str());

        ImGui::SeparatorText("View");
        ImGui::Checkbox("Activity graphs", &show_graphs);
        ImGui::Checkbox(seq_live ? "Colour task groups" : layout ? "Colour regions (nerve / A1 / mouth)" : "Highlight group A", &showA);
        ImGui::SameLine();
        ImGui::Checkbox("Auto-rotate", &autorot);
        if (ImGui::Button("Reset view")) cam = Cam();
        ImGui::TextDisabled("Right-click + hold on the sphere: stimulate");
        ImGui::TextDisabled("Left-drag: rotate   Wheel: zoom");
        ImGui::End();

        if (show_graphs) {
            int ww2, wh2; glfwGetWindowSize(win, &ww2, &wh2);
            ImGui::SetNextWindowPos(ImVec2((float)ww2 - 432, 12), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(420, 0), ImGuiCond_FirstUseEver);
            ImGui::Begin("Activity", &show_graphs);
            auto plot = [&](const char* label, const std::vector<float>& h, float cur, const char* unit, ImVec4 col) {
                float mx = 1.f; for (float v : h) mx = std::max(mx, v);
                char ov[64]; std::snprintf(ov, sizeof(ov), "%.1f %s", cur, unit);
                ImGui::PushStyleColor(ImGuiCol_PlotLines, col);
                ImGui::PlotLines(label, h.data(), HIST, hpos, ov, 0.f, mx * 1.15f, ImVec2(300, 60));
                ImGui::PopStyleColor();
            };
            ImGui::TextDisabled("last 12 s of simulated time, newest on the right");
            // correlation over the window shown: does one graph follow another? (-1..1)
            auto corr = [&](const std::vector<float>& a, const std::vector<float>& b) {
                double ma = 0, mb = 0; for (int i = 0; i < HIST; ++i) { ma += a[i]; mb += b[i]; }
                ma /= HIST; mb /= HIST;
                double sab = 0, saa = 0, sbb = 0;
                for (int i = 0; i < HIST; ++i) { const double x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
                return (saa > 0 && sbb > 0) ? sab / std::sqrt(saa * sbb) : 0.0;
            };
            if (EN > 0) {
                plot("Screen\nbrightness", h_bright, scr_bright * 100.f, "%", ImVec4(0.8f, 0.8f, 0.8f, 1));
                plot("Screen\nchange", h_chg, scr_change * 100.f, "% per frame", ImVec4(0.6f, 0.6f, 1.0f, 1));
                plot("Eye\nneurons", h_eye, cur_rate[0], "Hz", ImVec4(1.0f, 0.6f, 0.2f, 1));
            }
            if (AN > 0) {
                plot("Sound\nloudness", h_loud, cur_loud, "%", ImVec4(0.6f, 0.9f, 0.6f, 1));
                if (reward_ok) plot("Dopamine", h_da, (float)dopamine, "", ImVec4(0.3f, 1.0f, 0.5f, 1));
                plot("Ear\nneurons", h_ear, cur_rate[1], "Hz", ImVec4(0.3f, 0.9f, 0.4f, 1));
            }
            if (MN > 0) plot("Mouth\nneurons", h_mouth, cur_rate[2], "Hz", ImVec4(0.9f, 0.4f, 0.9f, 1));
            plot((EN + AN + MN) > 0 ? "Rest of\nE cells" : "E cells", h_rest, cur_rate[3], "Hz", ImVec4(0.35f, 0.6f, 1.0f, 1));
            plot("Inhibitory\ncells", h_inh, cur_rate[4], "Hz", ImVec4(1.0f, 0.35f, 0.35f, 1));
            if (AN > 0) {
                ImGui::SeparatorText("Does it follow the sound?  (correlation, -1..1)");
                ImGui::Text("ear neurons vs loudness      %+.2f", corr(h_ear, h_loud));
                ImGui::Text("rest of brain vs loudness    %+.2f", corr(h_rest, h_loud));
                if (MN > 0) ImGui::Text("mouth vs loudness            %+.2f", corr(h_mouth, h_loud));
            }
            if (EN > 0) {
                ImGui::SeparatorText("Does it follow the screen?  (correlation, -1..1)");
                ImGui::Text("eye neurons vs brightness   %+.2f", corr(h_eye, h_bright));
                ImGui::Text("rest of brain vs brightness %+.2f", corr(h_rest, h_bright));
                ImGui::Text("rest of brain vs change     %+.2f", corr(h_rest, h_chg));
                ImGui::SeparatorText("Screen vs what its eye neurons do");
                ImGui::Image((ImTextureID)(intptr_t)eye_tex, ImVec2(EW * 1.0f, EH * 1.0f));
                ImGui::SameLine();
                ImGui::Image((ImTextureID)(intptr_t)act_tex, ImVec2(EW * 1.0f, EH * 1.0f));
            }
            ImGui::End();
        }
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
#endif
        glfwSwapBuffers(win);

        ++frames;
        if (now - t_title > 0.5) {
            char title[256];
            std::snprintf(title, sizeof(title),
                          "Brain in a vat | %d neurons | t = %.1f s | %.3gx (actual %.2fx) | %.0f fps | "
                          "inhibition %.2f | drive %.2f | %s%s",
                          N, step * p.dt / 1000.0, speed, sim_acc / (now - t_title), frames / (now - t_title),
                          inh, drive, seq_status(), paused ? " | PAUSED" : "");
            glfwSetWindowTitle(win, title);
            disp_speed = sim_acc / (now - t_title); disp_fps = frames / (now - t_title);
            t_title = now; frames = 0; sim_acc = 0.0;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_poke_idx);
    if (eye_tex) glDeleteTextures(1, &eye_tex);
    if (act_tex) glDeleteTextures(1, &act_tex);
    cudaFree(d_cnt);
    if (ears_live) ears_stop();
    if (mouth_live) mouth_stop();
    cudaGraphicsUnregisterResource(glow_res);
#ifdef BRAIN_VIZ_UI
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
#endif
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
#endif  // BRAIN_VIZ

