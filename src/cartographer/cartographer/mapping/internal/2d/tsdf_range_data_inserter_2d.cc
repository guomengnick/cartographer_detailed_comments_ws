/*
 * Copyright 2018 The Cartographer Authors
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

#include "cartographer/mapping/internal/2d/tsdf_range_data_inserter_2d.h"

#include "cartographer/mapping/internal/2d/normal_estimation_2d.h"
#include "cartographer/mapping/internal/2d/ray_to_pixel_mask.h"

namespace cartographer {
namespace mapping {
namespace {

// Factor for subpixel accuracy of start and end point for ray casts.
constexpr int kSubpixelScale = 1000;
// Minimum distance between range observation and origin. Otherwise, range
// observations are discarded.
constexpr float kMinRangeMeters = 1e-6f;
const float kSqrtTwoPi = std::sqrt(2.0 * M_PI);
/*
* range_data.origin：local-> 此幀tracking匹配後的位置(XYZ)
* 
*/

void GrowAsNeeded(const sensor::RangeData& range_data,
                  const float truncation_distance, TSDF2D* const tsdf) {
  Eigen::AlignedBox2f bounding_box(range_data.origin.head<2>());
  for (const sensor::RangefinderPoint& hit : range_data.returns) {
    const Eigen::Vector3f direction =
        (hit.position - range_data.origin).normalized();
    const Eigen::Vector3f end_position =
        hit.position + truncation_distance * direction;
    bounding_box.extend(end_position.head<2>());
  }
  // Padding around bounding box to avoid numerical issues at cell boundaries.
  constexpr float kPadding = 1e-6f;
  tsdf->GrowLimits(bounding_box.min() - kPadding * Eigen::Vector2f::Ones());
  tsdf->GrowLimits(bounding_box.max() + kPadding * Eigen::Vector2f::Ones());
}

float GaussianKernel(const float x, const float sigma) {
  return 1.0 / (kSqrtTwoPi * sigma) * std::exp(-0.5 * x * x / (sigma * sigma));
}

std::pair<Eigen::Array2i, Eigen::Array2i> SuperscaleRay(
    const Eigen::Vector2f& begin, const Eigen::Vector2f& end,
    TSDF2D* const tsdf) {
  const MapLimits& limits = tsdf->limits();
  const double superscaled_resolution = limits.resolution() / kSubpixelScale;
  const MapLimits superscaled_limits(
      superscaled_resolution, limits.max(),
      CellLimits(limits.cell_limits().num_x_cells * kSubpixelScale,
                 limits.cell_limits().num_y_cells * kSubpixelScale));

  const Eigen::Array2i superscaled_begin =
      superscaled_limits.GetCellIndex(begin);
  const Eigen::Array2i superscaled_end = superscaled_limits.GetCellIndex(end);
  return std::make_pair(superscaled_begin, superscaled_end);
}

struct RangeDataSorter {
  RangeDataSorter(Eigen::Vector3f origin) { origin_ = origin.head<2>(); }
  bool operator()(const sensor::RangefinderPoint& lhs,
                  const sensor::RangefinderPoint& rhs) {
    const Eigen::Vector2f delta_lhs =
        (lhs.position.head<2>() - origin_).normalized();
    const Eigen::Vector2f delta_rhs =
        (rhs.position.head<2>() - origin_).normalized();
    if ((delta_lhs[1] < 0.f) != (delta_rhs[1] < 0.f)) {
      return delta_lhs[1] < 0.f;
    } else if (delta_lhs[1] < 0.f) {
      return delta_lhs[0] < delta_rhs[0];
    } else {
      return delta_lhs[0] > delta_rhs[0];
    }
  }

 private:
  Eigen::Vector2f origin_;
};

float ComputeRangeWeightFactor(float range, int exponent) {
  float weight = 0.f;
  if (std::abs(range) > kMinRangeMeters) {
    weight = 1.f / (std::pow(range, exponent));
  }
  return weight;
}
}  // namespace

proto::TSDFRangeDataInserterOptions2D CreateTSDFRangeDataInserterOptions2D(
    common::LuaParameterDictionary* parameter_dictionary) {
  proto::TSDFRangeDataInserterOptions2D options;
  options.set_truncation_distance(
      parameter_dictionary->GetDouble("truncation_distance"));
  options.set_maximum_weight(parameter_dictionary->GetDouble("maximum_weight"));
  options.set_update_free_space(
      parameter_dictionary->GetBool("update_free_space"));
  *options
       .mutable_normal_estimation_options() = CreateNormalEstimationOptions2D(
      parameter_dictionary->GetDictionary("normal_estimation_options").get());
  options.set_project_sdf_distance_to_scan_normal(
      parameter_dictionary->GetBool("project_sdf_distance_to_scan_normal"));
  options.set_update_weight_range_exponent(
      parameter_dictionary->GetInt("update_weight_range_exponent"));
  options.set_update_weight_angle_scan_normal_to_ray_kernel_bandwidth(
      parameter_dictionary->GetDouble(
          "update_weight_angle_scan_normal_to_ray_kernel_bandwidth"));
  options.set_update_weight_distance_cell_to_hit_kernel_bandwidth(
      parameter_dictionary->GetDouble(
          "update_weight_distance_cell_to_hit_kernel_bandwidth"));
  return options;
}

TSDFRangeDataInserter2D::TSDFRangeDataInserter2D(
    const proto::TSDFRangeDataInserterOptions2D& options)
    : options_(options) {}

// Casts a ray from origin towards hit for each hit in range data.
// If 'options.update_free_space' is 'true', all cells along the ray
// until 'truncation_distance' behind hit are updated. Otherwise, only the cells
// within 'truncation_distance' around hit are updated.

void TSDFRangeDataInserter2D::Insert(const sensor::RangeData& range_data,
                                     GridInterface* grid) const {
  const float truncation_distance =
      static_cast<float>(options_.truncation_distance());
  TSDF2D* tsdf = static_cast<TSDF2D*>(grid);
  GrowAsNeeded(range_data, truncation_distance, tsdf);

  // Compute normals if needed.
  bool scale_update_weight_angle_scan_normal_to_ray =
      options_.update_weight_angle_scan_normal_to_ray_kernel_bandwidth() != 0.f;//0.5
  sensor::RangeData sorted_range_data = range_data;
  std::vector<float> normals;
  //預設是 true , 0.5 ，都會進入
  if (options_.project_sdf_distance_to_scan_normal() ||
      scale_update_weight_angle_scan_normal_to_ray) {
    std::vector<sensor::RangefinderPoint> returns =
        sorted_range_data.returns.points();
    std::sort(returns.begin(), returns.end(),
              RangeDataSorter(sorted_range_data.origin));
    sorted_range_data.returns = sensor::PointCloud(std::move(returns));
    //預設參數會計算normal
    normals = EstimateNormals(sorted_range_data,
                              options_.normal_estimation_options());
  }

  const Eigen::Vector2f origin = sorted_range_data.origin.head<2>();
  for (size_t hit_index = 0; hit_index < sorted_range_data.returns.size();
       ++hit_index) {
    const Eigen::Vector2f hit =
        sorted_range_data.returns[hit_index].position.head<2>();
    const float normal = normals.empty()
                             ? std::numeric_limits<float>::quiet_NaN()
                             : normals[hit_index];
    InsertHit(options_, hit, origin, normal, tsdf);
  }
  tsdf->FinishUpdate();
}

// 对TSDF地图进行更新hit
//要寫入地圖的點： hit 、normal
//kuo hit: 遍歷的每一個障礙物點，方向是local 座標系下的tracking
// normal: 遍歷的每一個障礙物點的垂直方向
void TSDFRangeDataInserter2D::InsertHit(
    const proto::TSDFRangeDataInserterOptions2D& options,
    const Eigen::Vector2f& hit, const Eigen::Vector2f& origin, float normal,
    TSDF2D* tsdf) const {
  const Eigen::Vector2f ray = hit - origin;
  const float range = ray.norm();
  const float truncation_distance =
      static_cast<float>(options_.truncation_distance());
  if (range < truncation_distance) return;
  const float truncation_ratio = truncation_distance / range;
  const Eigen::Vector2f ray_begin =
      options_.update_free_space() ? origin
                                   : origin + (1.0f - truncation_ratio) * ray;
  const Eigen::Vector2f ray_end = origin + (1.0f + truncation_ratio) * ray;
  std::pair<Eigen::Array2i, Eigen::Array2i> superscaled_ray =
      SuperscaleRay(ray_begin, ray_end, tsdf);
  std::vector<Eigen::Array2i> ray_mask = RayToPixelMask(
      superscaled_ray.first, superscaled_ray.second, kSubpixelScale);

  // Precompute weight factors.
  //會將weight_factor_angle_ray_normal更新，但最後更新權重還是會用tsd 進入高斯核的值
  float weight_factor_angle_ray_normal = 1.f;
  if (options_.update_weight_angle_scan_normal_to_ray_kernel_bandwidth() !=//0.5
      0.f) {
    //negative_ray：需要乘負號，反向才會跟normal同方向
    const Eigen::Vector2f negative_ray = -ray;
    float angle_ray_normal =
        common::NormalizeAngleDifference(normal - common::atan2(negative_ray));
    //預設會用高斯核更新 weight
    weight_factor_angle_ray_normal = GaussianKernel(
        angle_ray_normal,
        options_.update_weight_angle_scan_normal_to_ray_kernel_bandwidth());//0.5
  }
  float weight_factor_range = 1.f;
  if (options_.update_weight_range_exponent() != 0) {//0.
    //預設不會用距離更新weight
    weight_factor_range = ComputeRangeWeightFactor(
        range, options_.update_weight_range_exponent());
  }

  // Update Cells.此hit 前後截斷距離的數個柵格
  for (const Eigen::Array2i& cell_index : ray_mask) {
    if (tsdf->CellIsUpdated(cell_index)) continue;
    Eigen::Vector2f cell_center = tsdf->limits().GetCellCenter(cell_index);
    float distance_cell_to_origin = (cell_center - origin).norm();
    float update_tsd = range - distance_cell_to_origin;
    if (options_.project_sdf_distance_to_scan_normal()) {//true
      //預設會更新初始值，將觀測時有多垂直於雷達加入考量
      float normal_orientation = normal;
      update_tsd = (cell_center - hit)
                       .dot(Eigen::Vector2f{std::cos(normal_orientation),
                                            std::sin(normal_orientation)});
    }
    update_tsd =
        common::Clamp(update_tsd, -truncation_distance, truncation_distance);
    float update_weight = weight_factor_range * weight_factor_angle_ray_normal;
    if (options_.update_weight_distance_cell_to_hit_kernel_bandwidth() != 0.f) {  //0.5
      //預設會用高斯核更新update_weight
      //距離中心點越遠，則權重遞減
      update_weight *= GaussianKernel(
          update_tsd,
          options_.update_weight_distance_cell_to_hit_kernel_bandwidth());
    }
    UpdateCell(cell_index, update_tsd, update_weight, tsdf);
  }
}

// TSDF地图栅格的更新, 分别更新tsd值与权重值
void TSDFRangeDataInserter2D::UpdateCell(const Eigen::Array2i& cell,
                                         float update_sdf, float update_weight,
                                         TSDF2D* tsdf) const {
  if (update_weight == 0.f) return;
  // 获取TSD值与权重值，  first是tsd初始-0.3 ，second是權重初始0，
  const std::pair<float, float> tsd_and_weight = tsdf->GetTSDAndWeight(cell);
  float updated_weight = tsd_and_weight.second + update_weight;
  float updated_sdf = (tsd_and_weight.first * tsd_and_weight.second +
                       update_sdf * update_weight) /
                      updated_weight;
  //如果大於10就用10代替
  updated_weight =
      std::min(updated_weight, static_cast<float>(options_.maximum_weight()));
  tsdf->SetCell(cell, updated_sdf, updated_weight);
}

}  // namespace mapping
}  // namespace cartographer
