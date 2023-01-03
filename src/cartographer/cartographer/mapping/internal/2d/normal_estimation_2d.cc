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

#include "cartographer/mapping/internal/2d/normal_estimation_2d.h"

namespace cartographer {
namespace mapping {
namespace {

float NormalTo2DAngle(const Eigen::Vector3f& v) {
  return std::atan2(v[1], v[0]);
}

// Estimate the normal of an estimation_point as the arithmetic mean of the the
// normals of the vectors from estimation_point to each point in the
// sample_window.
/*
returns：此幀障礙物點
estimation_point_index：要預測的障礙物點
sample_window_begin：要計算障礙物點的normal 起始點
sample_window_end：結束點
sensor_origin：此幀tracking 在local 座標系下的位置
*/
float EstimateNormal(const sensor::PointCloud& returns,
                     const size_t estimation_point_index,
                     const size_t sample_window_begin,
                     const size_t sample_window_end,
                     const Eigen::Vector3f& sensor_origin) {
  const Eigen::Vector3f& estimation_point =
      returns[estimation_point_index].position;
  
  //當hit 鄰近點的起始、結束點只間隔1或0時，（造成原因可能太遠障礙物離雷達太遠等等）
  if (sample_window_end - sample_window_begin < 2) {
    return NormalTo2DAngle(sensor_origin - estimation_point);
  }
  Eigen::Vector3f mean_normal = Eigen::Vector3f::Zero();
  const Eigen::Vector3f& estimation_point_to_observation =
      sensor_origin - estimation_point;
  for (size_t sample_point_index = sample_window_begin;
       sample_point_index < sample_window_end; ++sample_point_index) {
    if (sample_point_index == estimation_point_index) continue;
    const Eigen::Vector3f& sample_point = returns[sample_point_index].position;
    const Eigen::Vector3f& tangent = estimation_point - sample_point;
    Eigen::Vector3f sample_normal = {-tangent[1], tangent[0], 0.f};
    constexpr float kMinNormalLength = 1e-6f;
    if (sample_normal.norm() < kMinNormalLength) {
      continue;
    }
    // Ensure sample_normal points towards 'sensor_origin'.
    if (sample_normal.dot(estimation_point_to_observation) < 0) {
      sample_normal = -sample_normal;
    }
    sample_normal.normalize();
    mean_normal += sample_normal;
  }
  return NormalTo2DAngle(mean_normal);
}
}  // namespace

proto::NormalEstimationOptions2D CreateNormalEstimationOptions2D(
    common::LuaParameterDictionary* parameter_dictionary) {
  proto::NormalEstimationOptions2D options;
  options.set_num_normal_samples(
      parameter_dictionary->GetInt("num_normal_samples"));
  options.set_sample_radius(parameter_dictionary->GetDouble("sample_radius"));
  CHECK_GT(options.num_normal_samples(), 0);
  CHECK_GT(options.sample_radius(), 0.0);
  return options;
}
//譯：預測每個障礙物點的normal，假設雷達點的角度已經排好序（已排好序），
//且是在origin座標系（此幀點雲匹配後在local座標系下的tracking位置）下到每一個障礙物點

// Estimates the normal for each 'return' in 'range_data'.
// Assumes the angles in the range data returns are sorted with respect to
// the orientation of the vector from 'origin' to 'return'.
//一幀點雲數據：體素濾波的點 voxel_filter_size
//     origin ：tracking 匹配後位於local座標系下的位置
//     returns：從tracking 到障礙物的點
//     misses ：沒用到 
std::vector<float> EstimateNormals(
    const sensor::RangeData& range_data,
    const proto::NormalEstimationOptions2D& normal_estimation_options) {
  std::vector<float> normals;
  normals.reserve(range_data.returns.size());
  const size_t max_num_samples = normal_estimation_options.num_normal_samples();
  const float sample_radius = normal_estimation_options.sample_radius();
  
  //遍歷此幀障礙物點
  for (size_t current_point = 0; current_point < range_data.returns.size(); ++current_point) {
    const Eigen::Vector3f& hit = range_data.returns[current_point].position;
    size_t sample_window_begin = current_point;
    //設定sample_window_begin
    //sample_window_begin：初始為0，如果current_point為0，則sample_window_begin也為0
    for (; sample_window_begin > 0 &&
           current_point - sample_window_begin < max_num_samples / 2 &&
           (hit - range_data.returns[sample_window_begin - 1].position).norm() <
               sample_radius;
         --sample_window_begin) {
    }
    size_t sample_window_end = current_point;
    //設定sample_window_end
    //current_point:0，sample_window_begin:0，sample_window_end就為2
    //且遍歷點~最後點不得大於採樣半徑0.5
    for (;
         sample_window_end < range_data.returns.size() &&
         sample_window_end - current_point < ceil(max_num_samples / 2.0) + 1 &&//2-0 < 3
         (hit - range_data.returns[sample_window_end].position).norm() <  
             sample_radius;
         ++sample_window_end) {
    }
    const float normal_estimate =
        EstimateNormal(range_data.returns, current_point, sample_window_begin,
                       sample_window_end, range_data.origin);
    normals.push_back(normal_estimate);
  }
  return normals;
}

}  // namespace mapping
}  // namespace cartographer
