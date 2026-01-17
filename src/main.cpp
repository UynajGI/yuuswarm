#include <chrono>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

#include "Engine.hpp"
#include "ExperimentConfig.hpp"
#include "Integrator.hpp"
#include "Models.hpp"
#include "NpzWriter.hpp"

// --- 辅助函数：初始化状态 ---
MatrixType init_states(const ExperimentConfig& cfg) {
  std::mt19937 gen(cfg.seed);
  std::uniform_real_distribution<> pos_dist(0.0, cfg.L);  // 使用 cfg.L
  std::normal_distribution<> spin_dist(0.0, 1.0);

  MatrixType y(cfg.n, 2 * cfg.d + 2 * cfg.d_s);

  for (int i = 0; i < cfg.n; ++i) {
    // X
    for (int k = 0; k < cfg.d; ++k) y(i, k) = pos_dist(gen);
    // V
    for (int k = 0; k < cfg.d; ++k) y(i, cfg.d + k) = 0.0;
    // S
    for (int k = 0; k < cfg.d_s; ++k) y(i, 2 * cfg.d + k) = spin_dist(gen);
    // W
    for (int k = 0; k < cfg.d_s; ++k) y(i, 2 * cfg.d + cfg.d_s + k) = 0.0;
  }
  return y;
}

int main(int argc, char* argv[]) {
  auto start_time = std::chrono::high_resolution_clock::now();

  if (argc < 2) {
    std::cerr << "Usage: ./simulation <config.json> [--force]" << std::endl;
    return 1;
  }

  try {
    // --- 1. 加载配置 ---
    auto config = ExperimentConfig::from_file(argv[1]);

    // --- 2. 检查去重 ---
    bool force = (argc >= 3 && std::string(argv[2]) == "--force");
    if (!config.prepare_workspace(force)) {
      std::cout << "[SKIP] Experiment " << config.run_id << " already exists."
                << std::endl;
      return 0;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "[START] Job: " << config.job_name << std::endl;
    std::cout << "        ID : " << config.run_id << std::endl;
    std::cout << "        N  : " << config.n << ", L=" << config.L << std::endl;
    std::cout << "        Save Every: " << config.save_interval << " steps"
              << std::endl;
    std::cout << "        Chunk Size: " << config.chunk_size << " frames"
              << std::endl;
    std::cout << "========================================" << std::endl;

    // --- 3. 物理初始化 ---
    SystemCoeffs coeffs;
    coeffs.m = VectorType::Zero(config.n);
    coeffs.beta_m = VectorType::Ones(config.n);
    coeffs.ms = VectorType::Zero(config.n);
    coeffs.beta_s = VectorType::Ones(config.n);

    MatrixType v0 = MatrixType::Zero(config.n, config.d);
    MatrixType omega0 = MatrixType::Zero(config.n, config.d_s);
    MatrixType y0 = init_states(config);

    // --- 4. 实例化引擎 ---
    using ModelType = OriginalInteraction;
    ModelType model;
    Engine<ModelType> engine(config.n, config.d, config.d_s, model);

    // --- 5. 运行积分 (分批保存模式) ---
    std::cout << "[RUN ] Starting integration..." << std::endl;

    // 定义缓冲区
    std::vector<Scalar> t_buffer;
    std::vector<MatrixType> y_buffer;
    // 预分配稍大一点，防止 realloc
    t_buffer.reserve(config.chunk_size + 50);
    y_buffer.reserve(config.chunk_size + 50);

    int chunk_counter = 0;
    long long step_counter = 0;  // 防止溢出用 long long

    // --- 定义回调函数 (Lambda) ---
    // 这个函数会被 Integrator 在每一步成功后调用
    auto save_callback = [&](Scalar t, const MatrixType& y_curr) {
      // 1. 降采样逻辑：只保存符合间隔的步
      if (step_counter % config.save_interval == 0) {
        t_buffer.push_back(t);
        y_buffer.push_back(y_curr);
      }
      step_counter++;

      // 2. 缓冲区检查：是否已满？
      if (y_buffer.size() >= config.chunk_size) {
        // 写入硬盘
        NpzWriter::save_chunk(config, chunk_counter, t_buffer, y_buffer);

        // 清空缓冲区 (capacity 保持不变，效率高)
        t_buffer.clear();
        y_buffer.clear();

        chunk_counter++;
      }
    };

    // --- 启动积分器 ---
    // 注意：这里不再传 vector，而是传 callback
    Integrator::rk2_fixed(engine, 0.0, config.t_end, config.dt, y0, v0, omega0,
                          coeffs, config.params, save_callback);

    // --- 6. 收尾：保存剩余数据 ---
    // 循环结束后，buffer 里可能还有不足 chunk_size 的数据，必须存下来
    if (!y_buffer.empty()) {
      NpzWriter::save_chunk(config, chunk_counter, t_buffer, y_buffer);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << "[DONE] Finished in " << duration.count() << "s." << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "\n[FATAL ERROR] " << e.what() << std::endl;
    return 1;
  }

  return 0;
}