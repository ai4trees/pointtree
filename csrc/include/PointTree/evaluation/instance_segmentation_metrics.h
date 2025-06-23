#include <omp.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../type_aliases.h"

using namespace Eigen;

#ifndef INSTANCE_SEGMENTATION_METRICS_H
#define INSTANCE_SEGMENTATION_METRICS_H

namespace PointTree {

template <typename scalar_T>
double quantile(const ArrayX<scalar_T> data, double q) {
  if (data.size() == 0) {
    throw std::invalid_argument("Cannot compute quantile of empty vector.");
  }
  if (q < 0.0 || q > 1.0) {
    throw std::invalid_argument("Quantile must be between 0 and 1.");
  }

  std::vector<scalar_T> values(data.data(), data.data() + data.size());
  std::sort(values.begin(), values.end());

  double pos = q * (values.size() - 1);
  int64_t idx_below = static_cast<int64_t>(std::floor(pos));
  int64_t idx_above = static_cast<int64_t>(std::ceil(pos));

  if (idx_below == idx_above) {
    return values[idx_below];
  }

  double fraction = pos - idx_below;
  return static_cast<double>(values[idx_below] * (1.0 - fraction) + values[idx_above] * fraction);
}

template <typename scalar_T>
std::tuple<ArrayXX<scalar_T>, ArrayXX<scalar_T>, ArrayXX<scalar_T>> compute_instance_segmentation_metrics_per_partition(
    const RefArrayX3<scalar_T> xyz,
    const RefArrayXl target,
    const RefArrayXl prediction,
    const RefArrayXl matched_predicted_ids,
    const std::string& partition,
    int64_t start_instance_id,
    bool include_unmatched_instances,
    int64_t invalid_instance_id,
    int64_t num_partitions = 10,
    int num_workers = -1) {
  if (partition != "xy" && partition != "z") {
    throw std::invalid_argument("Invalid partition scheme: " + partition);
  }
  if (num_workers <= 0) {
    num_workers = omp_get_max_threads();
  }

  int num_points = xyz.rows();
  int num_target_ids = matched_predicted_ids.size();

  ArrayXX<scalar_T> iou =
      ArrayXX<scalar_T>::Constant(num_target_ids, num_partitions, std::numeric_limits<scalar_T>::quiet_NaN());
  ArrayXX<scalar_T> precision =
      ArrayXX<scalar_T>::Constant(num_target_ids, num_partitions, std::numeric_limits<scalar_T>::quiet_NaN());
  ArrayXX<scalar_T> recall =
      ArrayXX<scalar_T>::Constant(num_target_ids, num_partitions, std::numeric_limits<scalar_T>::quiet_NaN());

  ArrayXX<scalar_T> intersection_count = ArrayXX<scalar_T>::Constant(num_target_ids, num_partitions, 0);
  ArrayXX<scalar_T> prediction_count = ArrayXX<scalar_T>::Constant(num_target_ids, num_partitions, 0);
  ArrayXX<scalar_T> target_count = ArrayXX<scalar_T>::Constant(num_target_ids, num_partitions, 0);

  scalar_T partition_size = 1.0 / num_partitions;

  MatrixX2<scalar_T> xy = xyz(Eigen::all, {0, 1}).matrix();
  KDTree2<scalar_T>* kd_tree = new KDTree2<scalar_T>(2, std::cref(xy), 10 /* max leaf size */);

  // #pragma omp parallel for num_threads(num_workers)
  for (int64_t target_idx = 0; target_idx < num_target_ids; ++target_idx) {
    int target_id = start_instance_id + target_idx;
    int predicted_id = matched_predicted_ids[target_idx];

    if (predicted_id == invalid_instance_id && !include_unmatched_instances) {
      continue;
    }

    std::vector<int64_t> target_indices;
    for (int64_t i = 0; i < num_points; ++i) {
      if (target(i) == target_id) {
        target_indices.push_back(i);
      }
    }

    ArrayX3<scalar_T> tree_xyz = xyz(target_indices, Eigen::all);
    scalar_T min_z = tree_xyz.col(2).minCoeff();

    ArrayX<scalar_T> distance;  //  = ArrayX<scalar_T>::Zero(num_points);

    std::vector<nanoflann::ResultItem<int64_t, scalar_T>> search_result;

    if (partition == "xy") {
      scalar_T z_threshold = min_z + 0.30;
      std::vector<int64_t> lowest_point_indices;
      for (int64_t i = 0; i < tree_xyz.rows(); ++i) {
        if (tree_xyz(i, 2) <= z_threshold) {
          lowest_point_indices.push_back(i);
        }
      }

      Eigen::Array<scalar_T, 1, 2> position = tree_xyz(lowest_point_indices, {0, 1}).colwise().mean();

      ArrayX2<scalar_T> tree_xy_centered = tree_xyz.leftCols(2).rowwise() - position;
      ArrayX<scalar_T> distances_target = tree_xy_centered.rowwise().norm();

      scalar_T regularized_max = static_cast<scalar_T>(quantile(distances_target, 0.95));
      scalar_T regularized_max_squared = regularized_max * regularized_max;

      const size_t num_neighbors =
          kd_tree->index_->radiusSearch(position.data(), regularized_max_squared, search_result);

      std::vector<int64_t> neighbor_indices;
      distance = ArrayX<scalar_T>(num_neighbors);

      for (int64_t i = 0; i < num_neighbors; ++i) {
        neighbor_indices.push_back(search_result[i].first);
      }
      distance = (xy(neighbor_indices, Eigen::all).array().rowwise() - position).rowwise().norm() / regularized_max;

    } else if (partition == "z") {
      Eigen::Array<scalar_T, 1, 2> position = tree_xyz(Eigen::all, {0, 1}).colwise().mean();
      ArrayX<scalar_T> z_values = tree_xyz.col(2) - min_z;
      scalar_T regularized_max = static_cast<scalar_T>(quantile(z_values, 0.95));

      std::vector<int64_t> prediction_indices;
      for (int64_t i = 0; i < num_points; ++i) {
        if (prediction(i) == predicted_id) {
          prediction_indices.push_back(i);
        }
      }
      ArrayX2<scalar_T> prediction_xy = xyz(prediction_indices, {0, 1});

      scalar_T radius = (tree_xyz.leftCols(2).rowwise() - position).rowwise().norm().maxCoeff();
      radius = std::max(radius, (prediction_xy.rowwise() - position).rowwise().norm().maxCoeff()) + 1.0;
      scalar_T radius_squared = radius * radius;

      const size_t num_neighbors = kd_tree->index_->radiusSearch(position.data(), radius_squared, search_result);

      distance = ArrayX<scalar_T>(num_neighbors);

      for (int64_t i = 0; i < num_neighbors; ++i) {
        distance(i) = (xyz(search_result[i].first, 2) - min_z) / regularized_max;
      }
    }

    for (int64_t i = 0; i < search_result.size(); ++i) {
      int64_t idx = search_result[i].first;
      int64_t partition_idx = static_cast<int64_t>(std::floor(distance(i) / partition_size));

      if (partition_idx < 0 || partition_idx >= num_partitions) {
        continue;
      }

      if (target(idx) == target_id) {
        target_count(target_idx, partition_idx) += 1.0;
      }

      if (predicted_id != invalid_instance_id) {
        if (target(idx) == target_id && prediction(idx) == predicted_id) {
          intersection_count(target_idx, partition_idx) += 1.0;
        }

        if (prediction(idx) == predicted_id) {
          prediction_count(target_idx, partition_idx) += 1.0;
        }
      }
    }

    for (int64_t i = 0; i < num_partitions; ++i) {
      scalar_T union_count =
          target_count(target_idx, i) + prediction_count(target_idx, i) - intersection_count(target_idx, i);
      if (union_count > 0) {
        iou(target_idx, i) = intersection_count(target_idx, i) / union_count;
      }
      if (prediction_count(target_idx, i) > 0) {
        precision(target_idx, i) = intersection_count(target_idx, i) / prediction_count(target_idx, i);
      }
      if (target_count(target_idx, i) > 0) {
        recall(target_idx, i) = intersection_count(target_idx, i) / target_count(target_idx, i);
      }
    }
  }

  delete kd_tree;
  return {iou, precision, recall};
}

}  // namespace PointTree

#endif  // INSTANCE_SEGMENTATION_METRICS_H