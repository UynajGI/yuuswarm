#pragma once
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

#include "types.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

class ExperimentConfig {
 public:
  // --- 配置字段 ---
  std::string job_name = "default_job";
  std::string output_base_dir = "./results";

  int n = 100;
  int d = 2;
  int d_s = 2;
  double L = 10.0;
  double t_end = 10.0;
  double dt = 0.01;
  int seed = 42;
  ModelParams params;  // 物理参数

  // [新增] IO 控制参数
  int save_interval = 1;  // 默认为 1 (不跳帧，保存每一步)
  int chunk_size = 1000;  // 缓冲区大小 (每存 1000 帧写一次硬盘并清空内存)

  // --- 运行时生成的字段 ---
  std::string run_id;
  fs::path full_output_dir;

  // [修改] JSON 绑定宏：加入 save_interval 和 chunk_size
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(ExperimentConfig, job_name, output_base_dir, n,
                                 d, d_s, L, t_end, dt, seed, params,
                                 save_interval, chunk_size)

  // --- 工厂方法：从文件加载 ---
  static ExperimentConfig from_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
      throw std::runtime_error("Cannot open config file: " + path);
    json j = json::parse(f);
    ExperimentConfig cfg = j.get<ExperimentConfig>();
    cfg.initialize();
    return cfg;
  }

  // --- 初始化：生成 ID 和 路径 ---
  void initialize() {
    run_id = generate_hash_id();

    // 目录结构: results / job_name / hash_id /
    full_output_dir = fs::path(output_base_dir) / job_name / run_id;
  }

  // --- 核心：生成唯一哈希 ID ---
  // 注意：save_interval 改变会影响数据密度，建议参与哈希
  // chunk_size
  // 只是内存优化手段，不影响物理结果，通常不参与哈希（或者参与也可以，看你需求）
  std::string generate_hash_id() const {
    json key_params = {
        {"n", n},
        {"d", d},
        {"d_s", d_s},
        {"L", L},
        {"t_end", t_end},
        {"dt", dt},
        {"seed", seed},
        {"params", params},
        {"save_interval", save_interval}  // 采样频率不同视为不同实验
    };

    // 计算哈希
    std::string s = key_params.dump();
    std::size_t h = std::hash<std::string>{}(s);

    // 转为 16 进制字符串
    std::stringstream ss;
    ss << std::hex << h;
    return ss.str();
  }

  // --- 准备工作空间 ---
  // 返回 false 表示结果已存在（应该跳过），true 表示已创建新目录
  bool prepare_workspace(bool force_overwrite = false) {
    if (fs::exists(full_output_dir)) {
      // 检查是否存在分片数据（只要有第0片就算存在）
      bool has_data = fs::exists(full_output_dir / "simulation_part_0.npz");

      if (has_data && !force_overwrite) {
        return false;  // 已存在，且不强制覆盖
      }
    }
    fs::create_directories(full_output_dir);

    // 备份配置
    save_snapshot();
    return true;
  }

  void save_snapshot() const {
    json j = *this;
    std::ofstream o(full_output_dir / "config_snapshot.json");
    o << std::setw(4) << j << std::endl;
  }
};