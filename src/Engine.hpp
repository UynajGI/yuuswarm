#pragma once
#include <omp.h>

#include <iostream>

#include "types.hpp"

template <typename InteractionModel>
class Engine {
 public:
  int n, d, d_s;
  InteractionModel model;  // 实例化模型

  // 预分配临时内存，虽然OpenMP里需要私有变量，但在某些架构下可以优化
  // 这里主要存储全局的力矩阵
  MatrixType total_fx;
  MatrixType total_fs;

  Engine(int n, int d, int d_s, InteractionModel model_instance)
      : n(n), d(d), d_s(d_s), model(model_instance) {
    total_fx.resize(n, d);
    total_fs.resize(n, d_s);
  }

  // --- 归一化工具 ---
  static void normalize_inplace(Eigen::Ref<MatrixType> y, int n, int d,
                                int d_s) {
    int start_col = 2 * d;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      auto s = y.block(i, start_col, 1, d_s);
      Scalar norm = s.norm();
      if (norm > 1e-12) {
        s /= norm;
      }
    }
  }

  // --- 计算导数 dydt ---
  // t: 当前时间
  // y: 当前状态 [X, V, S, W]
  // dydt: 输出导数 (引用传递，避免返回大对象)
  void compute_derivatives(Scalar t, const Eigen::Ref<const MatrixType>& y,
                           const MatrixType& v0, const MatrixType& omega0,
                           const SystemCoeffs& coeffs,
                           const ModelParams& params,
                           Eigen::Ref<MatrixType> dydt) {
    // 1. 视图切片 (Zero-copy)
    auto X = y.block(0, 0, n, d);
    auto V = y.block(0, d, n, d);
    auto S = y.block(0, 2 * d, n, d_s);
    auto W = y.block(0, 2 * d + d_s, n, d_s);

    // 2. 清零力矩阵
    total_fx.setZero();
    total_fs.setZero();

// 3. 并行计算力 (Compute All Forces)
#pragma omp parallel
    {
      // 线程私有变量
      VectorType fx_i(d), fs_i(d_s);
      VectorType dv(d), ds(d_s);

#pragma omp for schedule(dynamic)
      for (int i = 0; i < n; ++i) {
        fx_i.setZero();
        fs_i.setZero();

        for (int j = 0; j < n; ++j) {
          if (i == j) continue;

          // 调用模板模型
          model(X.row(i), X.row(j), S.row(i), S.row(j), params, dv, ds);

          fx_i += dv;
          fs_i += ds;
        }

        // 写入全局矩阵 (i不同，无冲突)
        total_fx.row(i) = fx_i.transpose() / static_cast<Scalar>(n) + v0.row(i);
        total_fs.row(i) =
            fs_i.transpose() / static_cast<Scalar>(n) + omega0.row(i);
      }
    }

// 4. 计算 dydt
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      // --- Position ---
      if (coeffs.m[i] == 0.0) {
        dydt.block(i, 0, 1, d) = total_fx.row(i);
        dydt.block(i, d, 1, d).setZero();
      } else {
        dydt.block(i, 0, 1, d) = V.row(i);
        dydt.block(i, d, 1, d) =
            (total_fx.row(i) - coeffs.beta_m[i] * V.row(i)) / coeffs.m[i];
      }

      // --- Spin ---
      if (coeffs.ms[i] == 0.0) {
        VectorType s_curr = S.row(i);
        VectorType f_curr = total_fs.row(i);
        dydt.block(i, 2 * d, 1, d_s) =
            (f_curr - s_curr.dot(f_curr) * s_curr).transpose();
        dydt.block(i, 2 * d + d_s, 1, d_s).setZero();
      } else {
        dydt.block(i, 2 * d, 1, d_s) = W.row(i);
        dydt.block(i, 2 * d + d_s, 1, d_s) =
            (total_fs.row(i) - coeffs.beta_s[i] * W.row(i)) / coeffs.ms[i];
      }
    }
  }
};
