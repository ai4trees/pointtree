#include <omp.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <nanoflann.hpp>
#include <queue>
#include <tuple>
#include <vector>

#include "../type_aliases.h"

#ifndef COARSE_TO_FINE_ALGORITHM_H
#define COARSE_TO_FINE_ALGORITHM_H

namespace PointTree {

template <typename scalar_T>
ArrayXl grow_trees(
    RefArrayX3<scalar_T> tree_xyz,
    RefArrayXl instance_ids,
    RefArrayXl unique_instances_ids,
    RefArrayX<scalar_T> grid_origin,
    std::vector<ArrayXX<scalar_T>> crown_distance_fields,
    RefArrayXb seed_mask,
    scalar_T z_scale,
    int64_t num_neighbors_region_growing,
    scalar_T max_radius_region_growing,
    scalar_T grid_size_canopy_height_model,
    scalar_T multiplier_outside_coarse_border,
    int num_workers = -1) {
  if (num_workers <= 0) {
    num_workers = omp_get_max_threads();
  }

  int64_t num_points = tree_xyz.rows();
  int64_t num_instances = unique_instances_ids.rows();
  int64_t num_instances_region_growing = crown_distance_fields.size();
  scalar_T max_radius_region_growing_squared = max_radius_region_growing * max_radius_region_growing;
  scalar_T multiplier_outside_coarse_border_squared =
      multiplier_outside_coarse_border * multiplier_outside_coarse_border;

  if (seed_mask.sum() == 0) {
    return instance_ids;
  }

  num_neighbors_region_growing = std::min(num_neighbors_region_growing, num_points);

  std::vector<int64_t> growing_indices;
  std::set<int64_t> region_growing_instance_ids;

  for (int64_t i = 0; i < num_points; ++i) {
    if (seed_mask(i)) {
      growing_indices.push_back(i);
      region_growing_instance_ids.insert(instance_ids(i));
    } else if (instance_ids(i) == -1) {
      growing_indices.push_back(i);
    }
  }

  ArrayX3<scalar_T> growing_xyz = tree_xyz(growing_indices, Eigen::all);
  growing_xyz.col(2) *= z_scale;
  ArrayXl growing_instance_ids = instance_ids(growing_indices);

  ArrayXl point_indices = ArrayXl::LinSpaced(num_points, 0, num_points - 1);

  ArrayXl instance_id_mapping = ArrayXl::Constant(num_instances, -1);

  int64_t remapped_id = 0;
  for (int64_t i = 0; i < num_instances; ++i) {
    int64_t instance_id = unique_instances_ids(i);
    auto search = region_growing_instance_ids.find(instance_id);
    if (search != region_growing_instance_ids.end()) {
      instance_id_mapping(instance_id) = remapped_id;
      remapped_id += 1;
    }
  }

  MatrixX3<scalar_T> growing_xyz_mat = growing_xyz;
  KDTree3<scalar_T> kd_tree(3, std::cref(growing_xyz_mat), 10 /* max leaf size */);

  ArrayXXl neighbor_indices(growing_xyz.rows(), num_neighbors_region_growing);
  ArrayXX<scalar_T> squared_neighbor_dists(growing_xyz.rows(), num_neighbors_region_growing);

#pragma omp parallel for num_threads(num_workers)
  for (int64_t i = 0; i < growing_xyz.rows(); ++i) {
    std::vector<int64_t> knn_index(num_neighbors_region_growing);
    std::vector<scalar_T> knn_dist(num_neighbors_region_growing);
    ArrayX3<scalar_T> query_xyz = growing_xyz(i, Eigen::all);

    auto num_results =
        kd_tree.index_->knnSearch(query_xyz.data(), num_neighbors_region_growing, &knn_index[0], &knn_dist[0]);
    assert(num_results == num_neighbors_region_growing);
    neighbor_indices(i, Eigen::all) = Eigen::Map<ArrayXl>(knn_index.data(), num_neighbors_region_growing);
    squared_neighbor_dists(i, Eigen::all) = Eigen::Map<ArrayX<scalar_T>>(knn_dist.data(), num_neighbors_region_growing);
  }

  auto cmp = [](QueueElementType<scalar_T> left, QueueElementType<scalar_T> right) {
    return std::get<2>(left) > std::get<2>(right);
  };
  std::priority_queue<QueueElementType<scalar_T>, std::vector<QueueElementType<scalar_T>>, decltype(cmp)>
      priority_queue(cmp);

  for (int64_t i = 0; i < growing_indices.size(); ++i) {
    if (growing_instance_ids(i) != -1) {
      priority_queue.push(std::make_tuple(i, growing_instance_ids(i), -1));
    }
  }

  int64_t i = 0;
  ArrayXb visited = ArrayXb::Constant(growing_instance_ids.rows(), false);
  while (priority_queue.size() > 0) {
    QueueElementType<scalar_T> next_seed = priority_queue.top();
    priority_queue.pop();
    int64_t seed_index = std::get<0>(next_seed);
    int64_t instance_id = std::get<1>(next_seed);
    if (i % 10000 == 0) {
      std::cout << "Iteration " << i << ", seeds to process: " << priority_queue.size() << "." << std::endl;
    }
    if (visited(seed_index)) {
      continue;
    }
    growing_instance_ids(seed_index) = instance_id;
    visited(seed_index) = true;

    ArrayXl current_neighbor_instance_ids = growing_instance_ids(neighbor_indices(seed_index, Eigen::all));

    for (int64_t j = 0; j < num_neighbors_region_growing; ++j) {
      scalar_T squared_dist = squared_neighbor_dists(seed_index, j);
      int64_t neighbor_idx = neighbor_indices(seed_index, j);
      if (current_neighbor_instance_ids(j) == -1 && squared_dist <= max_radius_region_growing_squared) {
        int64_t grid_index_x =
            std::floor((growing_xyz(neighbor_idx, 0) - grid_origin(0)) / grid_size_canopy_height_model);
        int64_t grid_index_y =
            std::floor((growing_xyz(neighbor_idx, 1) - grid_origin(1)) / grid_size_canopy_height_model);
        scalar_T distance_to_crown_border =
            crown_distance_fields[instance_id_mapping(instance_id)](grid_index_x, grid_index_y);
        if (distance_to_crown_border > 0) {
          squared_dist *= multiplier_outside_coarse_border_squared;
        }
        priority_queue.push(std::make_tuple(neighbor_idx, instance_id, squared_dist));
      }
    }
    i += 1;
  }

  instance_ids(growing_indices) = growing_instance_ids;

  return instance_ids;
};

}  // namespace PointTree

#endif  // COARSE_TO_FINE_ALGORITHM_H