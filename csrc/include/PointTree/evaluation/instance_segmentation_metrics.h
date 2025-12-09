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
std::tuple<ArrayX<scalar_T>, ArrayX<scalar_T>, ArrayX<scalar_T>> compute_instance_segmentation_metrics(
    const RefArrayXl target,
    const RefArrayXl prediction,
    const RefArrayXl matched_predicted_ids,
    int64_t start_instance_id,
    int64_t invalid_instance_id,
    int num_workers = -1) {
  int64_t num_target_ids = matched_predicted_ids.size();
  int64_t num_points = target.size();

  ArrayX<scalar_T> iou = ArrayX<scalar_T>::Zero(num_target_ids);
  ArrayX<scalar_T> precision = ArrayX<scalar_T>::Zero(num_target_ids);
  ArrayX<scalar_T> recall = ArrayX<scalar_T>::Zero(num_target_ids);

  if (num_workers <= 0) {
    num_workers = omp_get_max_threads();
  }

#pragma omp parallel for num_threads(num_workers)
  for (int64_t target_idx = 0; target_idx < num_target_ids; ++target_idx) {
    int64_t target_id = start_instance_id + target_idx;
    int64_t predicted_id = matched_predicted_ids(target_idx);

    if (predicted_id == invalid_instance_id) {
      continue;
    }

    scalar_T intersection_count = 0.0;
    scalar_T prediction_count = 0.0;
    scalar_T target_count = 0.0;

    for (int64_t i = 0; i < num_points; ++i) {
      bool is_valid_target = (target(i) == target_id);
      bool is_valid_prediction = (prediction(i) == predicted_id);

      if (is_valid_target && is_valid_prediction) {
        intersection_count++;
      }
      if (is_valid_prediction) {
        prediction_count++;
      }
      if (is_valid_target) {
        target_count++;
      }
    }

    scalar_T union_count = target_count + prediction_count - intersection_count;

    if (union_count > 0) {
      iou(target_idx) = intersection_count / union_count;
    }
    if (prediction_count > 0) {
      precision(target_idx) = intersection_count / prediction_count;
    }
    if (target_count > 0) {
      recall(target_idx) = intersection_count / target_count;
    }
  }

  return {iou, precision, recall};
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

  VectorXd intervals = VectorXd::LinSpaced(num_partitions + 1, 0.0, 1.0);

#pragma omp parallel for num_threads(num_workers)
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
    ArrayX<scalar_T> distance;

    if (partition == "xy") {
      scalar_T z_threshold = min_z + 0.30;
      std::vector<int64_t> lowest_point_indices;
      for (int64_t i = 0; i < tree_xyz.rows(); ++i) {
        if (tree_xyz(i, 2) <= z_threshold) {
          lowest_point_indices.push_back(i);
        }
      }

      Eigen::Array<scalar_T, 1, 2> position = tree_xyz(lowest_point_indices, {0, 1}).colwise().mean();

      ArrayX<scalar_T> distance_target = (tree_xyz.leftCols(2).rowwise() - position).rowwise().norm();

      scalar_T regularized_max = static_cast<scalar_T>(quantile(distance_target, 0.95));

      distance = (xyz.leftCols(2).rowwise() - position).rowwise().norm();
      distance /= regularized_max;

    } else if (partition == "z") {
      ArrayX<scalar_T> z_values = tree_xyz.col(2) - min_z;
      scalar_T regularized_max = static_cast<scalar_T>(quantile(z_values, 0.95));
      distance = (xyz.col(2) - min_z) / (regularized_max);
    }

    ArrayX<scalar_T> intersection_count = ArrayX<scalar_T>::Zero(num_partitions);
    ArrayX<scalar_T> target_count = ArrayX<scalar_T>::Zero(num_partitions);
    ArrayX<scalar_T> prediction_count = ArrayX<scalar_T>::Zero(num_partitions);
    scalar_T partition_size = 1.0 / num_partitions;

    for (int64_t i = 0; i < num_points; ++i) {
      int64_t partition_idx = static_cast<int64_t>(std::floor(distance(i) / partition_size));

      if (partition_idx < 0 || partition_idx >= num_partitions) {
        continue;
      }

      if (target(i) == target_id) {
        target_count(partition_idx) += 1;
      }
      if (predicted_id != invalid_instance_id) {
        if (prediction(i) == predicted_id) {
          prediction_count(partition_idx) += 1;
        }
        if (target(i) == target_id && prediction(i) == predicted_id) {
          intersection_count(partition_idx) += 1;
        }
      }
    }

    ArrayX<scalar_T> union_count = target_count + prediction_count - intersection_count;

    for (int64_t i = 0; i < num_partitions; ++i) {
      if (union_count(i) > 0) {
        iou(target_idx, i) = intersection_count(i) / union_count(i);
      }
      if (target_count(i) > 0) {
        recall(target_idx, i) = intersection_count(i) / target_count(i);
      }
      if (prediction_count(i) > 0) {
        precision(target_idx, i) = intersection_count(i) / prediction_count(i);
      }
    }
  }

  return {iou, precision, recall};
}

}  // namespace PointTree

#endif  // INSTANCE_SEGMENTATION_METRICS_H