#pragma once
#include <iomanip>  // 必须引入，用于格式化文件名 (setw, setfill)
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ExperimentConfig.hpp"
#include "cnpy.h"
#include "types.hpp"

class NpzWriter {
 public:
  // --- [新增] 分片保存函数 ---
  static void save_chunk(const ExperimentConfig& cfg, int chunk_id,
                         const std::vector<double>& t,
                         const std::vector<MatrixType>& y_list) {
    if (t.empty()) return;

    // 1. 构造文件名: simulation_part_0000.npz
    // 使用 0000 补全，方便 python 按文件名排序读取
    std::stringstream ss;
    ss << "simulation_part_" << std::setw(5) << std::setfill('0') << chunk_id
       << ".npz";

    fs::path file_path = cfg.full_output_dir / ss.str();
    std::string filename = file_path.string();

    // 确保目录存在
    if (!fs::exists(cfg.full_output_dir)) {
      fs::create_directories(cfg.full_output_dir);
    }

    size_t steps = t.size();
    size_t rows = cfg.n;
    size_t cols = y_list[0].cols();

    // 2. 扁平化数据 (Copy Matrix to Flat Vector)
    std::vector<double> data_flat;
    data_flat.reserve(steps * rows * cols);
    for (const auto& mat : y_list) {
      data_flat.insert(data_flat.end(), mat.data(), mat.data() + mat.size());
    }

    // 3. 写入 time (覆盖模式 'w')
    cnpy::npz_save(filename, "time", t.data(), {steps}, "w");

    // 4. 写入 trajectory (追加模式 'a')
    cnpy::npz_save(filename, "trajectory", data_flat.data(),
                   {steps, rows, cols}, "a");

    // 5. 写入参数 (方便每个文件独立查看)
    std::vector<double> p_vec = {cfg.params.A, cfg.params.B, cfg.params.J,
                                 cfg.params.K};
    cnpy::npz_save(filename, "params", p_vec.data(), {p_vec.size()}, "a");

    std::cout << "[NPZ ] Saved chunk " << chunk_id << " to " << filename << " ("
              << steps << " steps)" << std::endl;
  }
};