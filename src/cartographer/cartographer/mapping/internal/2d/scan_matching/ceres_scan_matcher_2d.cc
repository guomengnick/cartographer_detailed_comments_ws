/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer/mapping/internal/2d/scan_matching/ceres_scan_matcher_2d.h"

#include <utility>
#include <vector>

#include "Eigen/Core"
#include "cartographer/common/internal/ceres_solver_options.h"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/occupied_space_cost_function_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/rotation_delta_cost_functor_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/translation_delta_cost_functor_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/tsdf_match_cost_function_2d.h"
#include "cartographer/transform/transform.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

proto::CeresScanMatcherOptions2D CreateCeresScanMatcherOptions2D(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::CeresScanMatcherOptions2D options;
  options.set_occupied_space_weight(
      parameter_dictionary->GetDouble("occupied_space_weight"));
  options.set_translation_weight(
      parameter_dictionary->GetDouble("translation_weight"));
  options.set_rotation_weight(
      parameter_dictionary->GetDouble("rotation_weight"));
  *options.mutable_ceres_solver_options() =
      common::CreateCeresSolverOptionsProto(
          parameter_dictionary->GetDictionary("ceres_solver_options").get());
  return options;
}

CeresScanMatcher2D::CeresScanMatcher2D(
    const proto::CeresScanMatcherOptions2D& options)
    : options_(options),
      ceres_solver_options_(
          common::CreateCeresSolverOptions(options.ceres_solver_options())) {
  ceres_solver_options_.linear_solver_type = ceres::DENSE_QR;
}

CeresScanMatcher2D::~CeresScanMatcher2D() {}

/**
 * @brief 基于Ceres的扫描匹配
 * 
 * @param[in] target_translation 预测出来的先验位置, 只有xy
 * @param[in] initial_pose_estimate (校正后的)先验位姿, 有xy与theta
 * @param[in] point_cloud 用于匹配的点云 点云的原点位于local坐标系原点
 * @param[in] grid 用于匹配的栅格地图
 * @param[out] pose_estimate 优化之后的位姿
 * @param[out] summary 
 */
void CeresScanMatcher2D::Match(const Eigen::Vector2d& target_translation,          //kuo:不會改的值，xy
                               const transform::Rigid2d& initial_pose_estimate,    //不會改的值，比上述多了旋轉
                               const sensor::PointCloud& point_cloud,              //200個點雲
                               const Grid2D& grid,                                 //柵格地圖
                               transform::Rigid2d* const pose_estimate,            //這個值為空，最後會對其賦值，這個值可以改，但是指向的東西不能改
                               ceres::Solver::Summary* const summary) const {
  //kuo：ceres吃的是double，這個數據是list，只要傳ceres_pose_estimate 就是指標了，會對其改值，solve後再給pose_estimate
  double ceres_pose_estimate[3] = {initial_pose_estimate.translation().x(),
                                   initial_pose_estimate.translation().y(),
                                   initial_pose_estimate.rotation().angle()};
  ceres::Problem problem;

  // 地图部分的残差
  CHECK_GT(options_.occupied_space_weight(), 0.);  //預設1
  switch (grid.GetGridType()) {
    case GridType::PROBABILITY_GRID:
      problem.AddResidualBlock(
          CreateOccupiedSpaceCostFunction2D(
              options_.occupied_space_weight() /
                  std::sqrt(static_cast<double>(point_cloud.size())),  //kuo:LaserScan每一個點的權重是  1/14
              point_cloud, grid),
          nullptr /* loss function */, ceres_pose_estimate);
      break;
    case GridType::TSDF:
      problem.AddResidualBlock(
          CreateTSDFMatchCostFunction2D(
              options_.occupied_space_weight() /
                  std::sqrt(static_cast<double>(point_cloud.size())),
              point_cloud, static_cast<const TSDF2D&>(grid)),
          nullptr /* loss function */, ceres_pose_estimate);
      break;
  }

  // 平移的残差
  CHECK_GT(options_.translation_weight(), 0.);
  problem.AddResidualBlock(
      TranslationDeltaCostFunctor2D::CreateAutoDiffCostFunction(
          options_.translation_weight(), target_translation), // 平移的目标值, 没有使用校准后的平移
      nullptr /* loss function */, ceres_pose_estimate);      // 平移的初值

  // 旋转的残差, 固定了角度不变
  CHECK_GT(options_.rotation_weight(), 0.);
  problem.AddResidualBlock(
      RotationDeltaCostFunctor2D::CreateAutoDiffCostFunction(
          options_.rotation_weight(), ceres_pose_estimate[2]), // 角度的目标值
      nullptr /* loss function */, ceres_pose_estimate);       // 角度的初值

  // 根据配置进行求解
  ceres::Solve(ceres_solver_options_, &problem, summary);

  *pose_estimate = transform::Rigid2d(
      {ceres_pose_estimate[0], ceres_pose_estimate[1]}, ceres_pose_estimate[2]);
}

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
