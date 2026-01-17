#pragma once
#include <algorithm>
#include <cmath>
#include <functional>  // 必须引入
#include <iostream>
#include <vector>

#include "Engine.hpp"
#include "types.hpp"

class Integrator {
 public:
  // --- 定义回调函数类型 ---
  // 参数: (当前时间 t, 当前状态 y)
  // 返回: void
  using Observer = std::function<void(Scalar, const MatrixType&)>;

  // --- RK2 固定步长 (使用 dt) ---
  template <typename EngineType>
  static void rk2_fixed(EngineType& engine, Scalar t0, Scalar tf, Scalar dt,
                        MatrixType& y0, const MatrixType& v0,
                        const MatrixType& omega0, const SystemCoeffs& coeffs,
                        const ModelParams& params,
                        Observer observer) {  // <--- 改动: 接受回调

    Scalar t = t0;
    MatrixType y = y0;

    EngineType::normalize_inplace(y, engine.n, engine.d, engine.d_s);

    // 1. 记录初始状态
    observer(t, y);

    int steps = std::ceil((tf - t0) / dt);

    MatrixType k1(y.rows(), y.cols());
    MatrixType k2(y.rows(), y.cols());
    MatrixType y_temp(y.rows(), y.cols());

    for (int i = 0; i < steps; ++i) {
      // k1 = f(t, y)
      engine.compute_derivatives(t, y, v0, omega0, coeffs, params, k1);

      // k2 = f(t+dt, y + dt*k1)
      y_temp = y + dt * k1;
      engine.compute_derivatives(t + dt, y_temp, v0, omega0, coeffs, params,
                                 k2);

      // y_next
      y += 0.5 * dt * (k1 + k2);
      t += dt;

      EngineType::normalize_inplace(y, engine.n, engine.d, engine.d_s);

      // 2. 记录当前步状态
      observer(t, y);
    }
    y0 = y;
  }

  // --- RK23 自适应步长 (使用 h) ---
  template <typename EngineType>
  static void rk23_adaptive(EngineType& engine, Scalar t0, Scalar tf,
                            Scalar dt_init, MatrixType& y0,
                            const MatrixType& v0, const MatrixType& omega0,
                            const SystemCoeffs& coeffs,
                            const ModelParams& params,
                            Observer observer) {  // <--- 改动: 接受回调

    Scalar t = t0;
    MatrixType y = y0;
    Scalar h = dt_init;

    Scalar rtol = 1e-3;
    Scalar atol = 1e-6;
    Scalar h_min = 1e-6;
    Scalar h_max = 0.5;

    EngineType::normalize_inplace(y, engine.n, engine.d, engine.d_s);

    // 1. 记录初始状态
    observer(t, y);

    MatrixType k1(y.rows(), y.cols());
    MatrixType k2(y.rows(), y.cols());
    MatrixType k3(y.rows(), y.cols());
    MatrixType y2(y.rows(), y.cols());  // 2阶解
    MatrixType y3(y.rows(), y.cols());  // 3阶解 (未使用, 仅作参考)
    MatrixType y_temp(y.rows(), y.cols());

    while (t < tf - 1e-12) {
      // 确保最后一步刚好到 tf
      if (t + h > tf) h = tf - t;

      // K1
      engine.compute_derivatives(t, y, v0, omega0, coeffs, params, k1);

      // K2
      y_temp = y + 0.5 * h * k1;
      engine.compute_derivatives(t + 0.5 * h, y_temp, v0, omega0, coeffs,
                                 params, k2);

      // K3
      y_temp = y + h * k2;  // Bogacki-Shampine 形式: y + 0.75*h*k2
                            // (此处保留你的逻辑 h*k2)
      engine.compute_derivatives(t + h, y_temp, v0, omega0, coeffs, params, k3);

      // 构造解
      // y2 (embedded low order solution) = y + h * (0.5*k1 + 0.5*k2)
      y2 = y + h * (0.5 * k1 + 0.5 * k2);

      // y_candidate (high order solution)
      MatrixType y_candidate = y + h * (k1 / 6.0 + 2.0 * k2 / 3.0 + k3 / 6.0);

      // 误差估算
      Scalar error_sq = (y_candidate - y2).squaredNorm();
      Scalar error = std::sqrt(error_sq / (y.size()));

      Scalar tol = atol + rtol * std::max(y.norm(), y_candidate.norm()) /
                              std::sqrt(y.size());

      if (error <= tol) {
        // --- Step Accepted ---
        t += h;
        y = y_candidate;
        EngineType::normalize_inplace(y, engine.n, engine.d, engine.d_s);

        // 2. 记录当前步 (关键修改点)
        observer(t, y);

        Scalar s =
            (error < 1e-15) ? 2.0 : 0.9 * std::pow(tol / error, 1.0 / 3.0);
        h = std::min(h_max, h * s);
      } else {
        // --- Step Rejected ---
        Scalar s =
            (error < 1e-15) ? 0.5 : 0.9 * std::pow(tol / error, 1.0 / 3.0);
        h = std::max(h_min, h * s);

        // 强制步进 (避免步长过小死循环)
        if (h <= h_min) {
          t += h;
          y = y_candidate;
          EngineType::normalize_inplace(y, engine.n, engine.d, engine.d_s);

          // 强制步进也需要记录
          observer(t, y);
        }
      }
    }
    y0 = y;
  }
};