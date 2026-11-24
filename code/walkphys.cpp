// walkphys.cpp - the walking creatures' bodies, simulated with Box2D (v3).
#include "walkphys.h"
#include <box2d/box2d.h>
#include <algorithm>
#include <cmath>
#ifdef _WIN32
#pragma comment(lib, "box2d.lib")
#endif

namespace {
// body plan (metres, kilograms per square metre)
const float TORSO_HW = 0.50f, TORSO_HH = 0.10f;      // half width / half height
const float LEG_HW   = 0.05f, THIGH_HH = 0.19f, SHIN_HH = 0.19f;
const float HIP_X    = 0.40f;                        // hips at -0.40 (back) and +0.40 (front)
const float STAND_Y  = 2.f * THIGH_HH + 2.f * SHIN_HH + TORSO_HH + 0.02f;   // torso centre height
const float DENSITY  = 8.0f, FRICTION = 0.9f;
const float HIP_TORQUE = 70.f, KNEE_TORQUE = 50.f;
const float JOINT_HZ = 240.f, JOINT_DAMP = 5.f;
const int   SUBSTEPS = 8;
const uint64_t CAT_GROUND = 0x1, CAT_CREATURE = 0x2;
}  // namespace

struct WalkWorld::Impl {
    b2WorldId world;
    struct Creature {
        b2BodyId body[WALK_PARTS];
        b2JointId joint[WALK_JOINTS];
        bool alive = false;
    };
    std::vector<Creature> c;

    b2BodyId make_box(float x, float y, float hw, float hh) {
        b2BodyDef bd = b2DefaultBodyDef();
        bd.type = b2_dynamicBody;
        bd.position.x = x; bd.position.y = y;
        bd.angularDamping = 0.5f;
        b2BodyId b = b2CreateBody(world, &bd);
        b2ShapeDef sd = b2DefaultShapeDef();
        sd.density = DENSITY;
        sd.material.friction = FRICTION;
        sd.filter.categoryBits = CAT_CREATURE;
        sd.filter.maskBits = CAT_GROUND;             // creatures only touch the ground
        b2Polygon box = b2MakeBox(hw, hh);
        b2CreatePolygonShape(b, &sd, &box);
        return b;
    }
    b2JointId hinge(b2BodyId a, b2Vec2 la, b2BodyId b, b2Vec2 lb, float lo, float hi, float torque) {
        b2RevoluteJointDef jd = b2DefaultRevoluteJointDef();
        jd.base.bodyIdA = a; jd.base.bodyIdB = b;
        jd.base.localFrameA.p = la; jd.base.localFrameA.q = b2Rot_identity;
        jd.base.localFrameB.p = lb; jd.base.localFrameB.q = b2Rot_identity;
        jd.enableLimit = true; jd.lowerAngle = lo; jd.upperAngle = hi;
        jd.enableMotor = true; jd.maxMotorTorque = torque; jd.motorSpeed = 0.f;
        jd.base.constraintHertz = JOINT_HZ;            // stiff joints: limits hold against the muscles
        jd.base.constraintDampingRatio = JOINT_DAMP;
        return b2CreateRevoluteJoint(world, &jd);
    }
    void build(int i) {
        Creature& k = c[i];
        if (k.alive) for (int p = 0; p < WALK_PARTS; ++p) b2DestroyBody(k.body[p]);   // joints go with them
        const float y0 = STAND_Y;
        k.body[0] = make_box(0.f, y0, TORSO_HW, TORSO_HH);
        for (int leg = 0; leg < 2; ++leg) {
            const float hx = leg == 0 ? -HIP_X : HIP_X;
            const float ty = y0 - TORSO_HH - THIGH_HH, sy = ty - THIGH_HH - SHIN_HH;
            k.body[1 + 2 * leg] = make_box(hx, ty, LEG_HW, THIGH_HH);
            k.body[2 + 2 * leg] = make_box(hx, sy, LEG_HW, SHIN_HH);
            k.joint[2 * leg] = hinge(k.body[0], b2Vec2{ hx, -TORSO_HH }, k.body[1 + 2 * leg], b2Vec2{ 0.f, THIGH_HH },
                                     joint_lower(2 * leg), joint_upper(2 * leg), HIP_TORQUE);
            k.joint[2 * leg + 1] = hinge(k.body[1 + 2 * leg], b2Vec2{ 0.f, -THIGH_HH }, k.body[2 + 2 * leg], b2Vec2{ 0.f, SHIN_HH },
                                         joint_lower(2 * leg + 1), joint_upper(2 * leg + 1), KNEE_TORQUE);
        }
        k.alive = true;
    }
};

// knees bend one way only, like a human's: the foot folds back (negative = shin turns clockwise)
float WalkWorld::joint_lower(int j) { return (j & 1) ? -2.0f : -1.1f; }
float WalkWorld::joint_upper(int j) { return (j & 1) ?  0.0f :  1.1f; }

WalkWorld::WalkWorld(int creatures) : impl_(new Impl), n_(creatures) {
    b2WorldDef wd = b2DefaultWorldDef();
    wd.gravity = b2Vec2{ 0.f, -9.81f };
    impl_->world = b2CreateWorld(&wd);
    b2BodyDef gd = b2DefaultBodyDef();                 // ground: a long flat strip at y = 0
    gd.position.x = 0.f; gd.position.y = -0.5f;
    b2BodyId ground = b2CreateBody(impl_->world, &gd);
    b2ShapeDef sd = b2DefaultShapeDef();
    sd.material.friction = FRICTION;
    sd.filter.categoryBits = CAT_GROUND;
    sd.filter.maskBits = CAT_CREATURE;
    b2Polygon g = b2MakeBox(5000.f, 0.5f);
    b2CreatePolygonShape(ground, &sd, &g);
    impl_->c.resize(creatures);
    for (int i = 0; i < creatures; ++i) impl_->build(i);
}

WalkWorld::~WalkWorld() { b2DestroyWorld(impl_->world); delete impl_; }

void WalkWorld::set_motors(int c, const float command[WALK_JOINTS]) {
    for (int j = 0; j < WALK_JOINTS; ++j) {
        const float u = std::max(-1.f, std::min(1.f, command[j]));
        b2RevoluteJoint_SetTargetMotorSpeed(impl_->c[c].joint[j], u * MAX_SPEED);
    }
}

void WalkWorld::step(float dt) { b2World_Step(impl_->world, dt, SUBSTEPS); }

void WalkWorld::reset(int c) { impl_->build(c); }

WalkState WalkWorld::state(int c) const {
    const Impl::Creature& k = impl_->c[c];
    WalkState s{};
    const auto p = b2Body_GetPosition(k.body[0]);
    const b2Vec2 v = b2Body_GetLinearVelocity(k.body[0]);
    s.x = (float)p.x; s.y = (float)p.y;
    s.angle = b2Rot_GetAngle(b2Body_GetRotation(k.body[0]));
    s.vx = v.x; s.vy = v.y; s.omega = b2Body_GetAngularVelocity(k.body[0]);
    for (int j = 0; j < WALK_JOINTS; ++j) {
        s.joint_angle[j] = b2RevoluteJoint_GetCurrentAngle(k.joint[j]);
        const int pa = j < 2 ? (j == 0 ? 0 : 1) : (j == 2 ? 0 : 3), pb = j + 1;
        s.joint_speed[j] = b2Body_GetAngularVelocity(k.body[pb]) - b2Body_GetAngularVelocity(k.body[pa]);
    }
    const float hh[WALK_PARTS] = { TORSO_HH, THIGH_HH, SHIN_HH, THIGH_HH, SHIN_HH };
    const float hw[WALK_PARTS] = { TORSO_HW, LEG_HW, LEG_HW, LEG_HW, LEG_HW };
    for (int q = 0; q < WALK_PARTS; ++q) {
        const auto bp = b2Body_GetPosition(k.body[q]);
        s.part[q] = WalkPart{ (float)bp.x, (float)bp.y, b2Rot_GetAngle(b2Body_GetRotation(k.body[q])), hw[q], hh[q] };
    }
    for (int leg = 0; leg < 2; ++leg) {                // shin tip height
        const auto tip = b2Body_GetWorldPoint(k.body[2 + 2 * leg], b2Vec2{ 0.f, -SHIN_HH });
        s.foot_down[leg] = tip.y < 0.03;
    }
    s.fallen = s.y < TORSO_HH + 0.08f;             // torso down on the ground (tilt is judged by the caller)
    return s;
}
