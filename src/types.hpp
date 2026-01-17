#pragma once
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using Scalar = double;
using MatrixType =
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorType = Eigen::VectorXd;

// 1. 物理参数 (纯数据 Struct)
struct ModelParams {
  double A = 1.0;
  double B = 1.0;
  double J = 0.5;
  double K = 0.1;
  double epsilon = 1.0;
  double sigma = 1.0;
  double eta = 1.0;
};

// 2. 宏：实现 Struct <-> JSON 的自动转换
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ModelParams, A, B, J, K, epsilon, sigma, eta)

struct SystemCoeffs {
  VectorType m, beta_m, ms, beta_s;
};