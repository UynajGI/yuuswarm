#pragma once
#include <cmath>

#include "types.hpp"

// === 模型 A: 原始 Attraction-Repulsion ===
struct OriginalInteraction {
  // 稍微放宽一点 EPS，防止极其接近时的数值爆炸
  static constexpr Scalar EPS = 1e-12;

  inline void operator()(const Eigen::Ref<const VectorType>& Xi,
                         const Eigen::Ref<const VectorType>& Xj,
                         const Eigen::Ref<const VectorType>& Si,
                         const Eigen::Ref<const VectorType>& Sj,
                         const ModelParams& p, Eigen::Ref<VectorType> f_pos,
                         Eigen::Ref<VectorType> f_spin) const {
    VectorType rij = Xj - Xi;

    // 1. 获取 r^2 (dist2)
    Scalar dist2 = rij.squaredNorm();

    // 保护：避免除以 0
    if (dist2 < EPS) {
      f_pos.setZero();
      f_spin.setZero();
      return;
    }

    // 2. 计算 1/r (inv_dist)
    // 技巧：你的 CMake 开启了 -ffast-math
    // 编译器会自动把 1.0 / sqrt(x) 优化为 CPU 的 RSQRT 指令 (速度快几倍)
    Scalar inv_dist = 1.0 / std::sqrt(dist2);

    // 3. 计算 1/r^2 (inv_dist2)
    // 直接用 inv_dist * inv_dist，比 1.0/dist2 快 (避免除法)
    Scalar inv_dist2 = inv_dist * inv_dist;

    // --- 计算位置力 ---
    Scalar dot_s = Si.dot(Sj);
    Scalar mag_att = p.A + p.J * dot_s;

    // 原始公式: F = (Att - B/r) * n_ij
    // 优化公式: F = (Att/r - B/r^2) * rij
    // 这样就把除法全部变成了标量乘法
    Scalar f_scalar = mag_att * inv_dist - p.B * inv_dist2;

    f_pos = f_scalar * rij;

    // --- 计算自旋力 ---
    // F_spin = (K/r) * Proj
    VectorType proj_sj = Sj - dot_s * Si;
    f_spin = (p.K * inv_dist) * proj_sj;
  }
};
// === 模型 B: LJ + Vicsek ===
struct LennardJonesVicsek {
  inline void operator()(const Eigen::Ref<const VectorType>& Xi,
                         const Eigen::Ref<const VectorType>& Xj,
                         const Eigen::Ref<const VectorType>& Si,
                         const Eigen::Ref<const VectorType>& Sj,
                         const VectorType& params, Eigen::Ref<VectorType> f_pos,
                         Eigen::Ref<VectorType> f_spin) const {
    Scalar epsilon = params[0];
    Scalar sigma = params[1];
    Scalar eta = params[2];

    VectorType rij = Xj - Xi;
    Scalar r2 = rij.squaredNorm();

    if (r2 > 9.0 * sigma * sigma) {
      f_pos.setZero();
      f_spin.setZero();
      return;
    }

    Scalar r6 = std::pow(sigma * sigma / r2, 3);
    Scalar r12 = r6 * r6;

    Scalar f_mag = 24.0 * epsilon * (2.0 * r12 - r6) / r2;
    f_pos = f_mag * rij;
    f_spin = eta * Sj;
  }
};