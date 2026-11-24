// walkphys.h - the walking creatures' bodies (2D physics with Box2D). Plain C++.
//
// Each creature, seen from the side: a torso and two legs (back and front), each leg a thigh
// and a shin. Four joints, each driven by a motor: back hip, back knee, front hip, front knee.
// Creatures share the ground but never touch each other.
#pragma once
#include <vector>

constexpr int WALK_JOINTS = 4;     // back hip, back knee, front hip, front knee
constexpr int WALK_PARTS  = 5;     // torso, back thigh, back shin, front thigh, front shin

struct WalkPart  { float x, y, angle, half_w, half_h; };   // for drawing
struct WalkState {
    float x, y, angle;             // torso position (m) and tilt (rad, 0 = level)
    float vx, vy, omega;           // torso velocity (m/s) and spin (rad/s)
    float joint_angle[WALK_JOINTS];
    float joint_speed[WALK_JOINTS];
    bool  foot_down[2];            // back / front shin tip touching the ground
    bool  fallen;                  // torso lying on the ground
    WalkPart part[WALK_PARTS];
};

class WalkWorld {
public:
    explicit WalkWorld(int creatures);
    ~WalkWorld();
    WalkWorld(const WalkWorld&) = delete;
    WalkWorld& operator=(const WalkWorld&) = delete;

    int  count() const { return n_; }
    // motor command per joint, -1 .. 1 (sets the joint motor's target speed)
    void set_motors(int c, const float command[WALK_JOINTS]);
    void step(float dt);                      // advance all creatures by dt seconds
    void reset(int c);                        // stand creature c up again at x = 0
    WalkState state(int c) const;

    static float joint_lower(int j);
    static float joint_upper(int j);
    static constexpr float MAX_SPEED = 8.0f;  // rad/s at command = 1

private:
    struct Impl;
    Impl* impl_;
    int n_;
};
