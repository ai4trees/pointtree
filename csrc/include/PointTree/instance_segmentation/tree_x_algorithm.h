#include <omp.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <nanoflann.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../operations/points_in_ellipse.h"
#include "../type_aliases.h"

#ifndef TREE_X_ALGORITHM_H
#define TREE_X_ALGORITHM_H

namespace PointTree {

template <typename scalar_T>
ArrayX2<scalar_T> compute_layer_bounds(
    scalar_T start_z, int64_t num_layers, scalar_T layer_height, scalar_T layer_overlap, int num_workers = -1) {
  if (num_workers <= 0) {
    num_workers = omp_get_max_threads();
  }

  ArrayX2<scalar_T> layer_bounds;
  layer_bounds.resize(num_layers, 2);

#pragma omp parallel for num_threads(num_workers)
  for (int64_t layer = 0; layer < num_layers; ++layer) {
    if (layer == 0) {
      layer_bounds(layer, 0) = start_z;
    } else {
      layer_bounds(layer, 0) = start_z + layer * (layer_height - layer_overlap);
    }
    layer_bounds(layer, 1) = layer_bounds(layer, 0) + layer_height;
  }

  return layer_bounds;
}

template <typename scalar_T>
std::tuple<ArrayX2<scalar_T>, ArrayXl> collect_inputs_trunk_layers_preliminary_fitting(
    RefArrayX3<scalar_T> trunk_layer_xyz,
    RefArrayXl cluster_labels,
    RefArrayXl unique_cluster_labels,
    scalar_T trunk_search_min_z,
    int64_t trunk_search_circle_fitting_num_layers,
    scalar_T trunk_search_circle_fitting_layer_height,
    scalar_T trunk_search_circle_fitting_layer_overlap,
    int64_t trunk_search_circle_fitting_min_points = 0,
    int num_workers = 1) {
  if (num_workers <= 0) {
    num_workers = omp_get_max_threads();
  }

  if (trunk_layer_xyz.rows() != cluster_labels.rows()) {
    throw std::invalid_argument("The length of trunk_layer_xyz and cluster_labels must be equal.");
  }

  auto num_labels = unique_cluster_labels.rows();
  auto num_layers = trunk_search_circle_fitting_num_layers;
  std::vector<std::vector<int64_t>> batch_indices(num_labels * num_layers);
  ArrayXl batch_lengths = ArrayXl::Constant(num_labels * num_layers, 0);

  ArrayX2<scalar_T> layer_bounds = compute_layer_bounds<scalar_T>(
      trunk_search_min_z, trunk_search_circle_fitting_num_layers, trunk_search_circle_fitting_layer_height,
      trunk_search_circle_fitting_layer_overlap, num_workers);

#pragma omp parallel for num_threads(num_workers)
  for (int64_t idx = 0; idx < num_labels * num_layers; ++idx) {
    int64_t label = idx / num_layers;
    int64_t layer = idx % num_layers;
    std::vector<int64_t> current_batch_indices;
    for (int64_t i = 0; i < trunk_layer_xyz.rows(); ++i) {
      if (cluster_labels(i) == unique_cluster_labels(label) && trunk_layer_xyz(i, 2) >= layer_bounds(layer, 0) &&
          trunk_layer_xyz(i, 2) <= layer_bounds(layer, 1)) {
        current_batch_indices.push_back(i);
      }
    }
    if (current_batch_indices.size() >= trunk_search_circle_fitting_min_points) {
      batch_indices[idx] = current_batch_indices;
      batch_lengths[idx] = current_batch_indices.size();
    }
  }

  ArrayXl selected_indices(batch_lengths.sum());
  int64_t start_idx = 0;
  for (int64_t i = 0; i < batch_indices.size(); ++i) {
    selected_indices(Eigen::seqN(start_idx, batch_lengths(i))) =
        Eigen::Map<ArrayXl>(batch_indices[i].data(), batch_indices[i].size());
    start_idx += batch_lengths(i);
  }

  return std::make_tuple(trunk_layer_xyz(selected_indices, {0, 1}), batch_lengths);
}

template <typename scalar_T>
std::tuple<ArrayX2<scalar_T>, ArrayXl, ArrayX<scalar_T>> collect_inputs_trunk_layers_exact_fitting(
    RefArrayX3<scalar_T> trunk_layer_xyz,
    RefArrayX5<scalar_T> preliminary_layer_circles_or_ellipses,
    scalar_T trunk_search_min_z,
    int64_t trunk_search_circle_fitting_num_layers,
    scalar_T trunk_search_circle_fitting_layer_height,
    scalar_T trunk_search_circle_fitting_layer_overlap,
    scalar_T trunk_search_circle_fitting_switch_buffer_threshold,
    scalar_T trunk_search_circle_fitting_small_buffer_width,
    scalar_T trunk_search_circle_fitting_large_buffer_width,
    int64_t trunk_search_circle_fitting_min_points = 0,
    int num_workers = -1) {
  if (num_workers <= 0) {
    num_workers = omp_get_max_threads();
  }

  auto num_layers = trunk_search_circle_fitting_num_layers;
  auto num_labels = preliminary_layer_circles_or_ellipses.rows() / num_layers;
  std::vector<std::vector<int64_t>> batch_indices(num_labels * num_layers);
  ArrayXl batch_lengths = ArrayXl::Constant(preliminary_layer_circles_or_ellipses.rows(), 0);

  ArrayX2<scalar_T> layer_bounds = compute_layer_bounds<scalar_T>(
      trunk_search_min_z, trunk_search_circle_fitting_num_layers, trunk_search_circle_fitting_layer_height,
      trunk_search_circle_fitting_layer_overlap, num_workers);

#pragma omp parallel for num_threads(num_workers)
  for (int64_t layer = 0; layer < num_layers; ++layer) {
    std::vector<int64_t> layer_point_indices;
    for (int64_t i = 0; i < trunk_layer_xyz.rows(); ++i) {
      if (trunk_layer_xyz(i, 2) >= layer_bounds(layer, 0) && trunk_layer_xyz(i, 2) <= layer_bounds(layer, 1)) {
        layer_point_indices.push_back(i);
      }
    }

    MatrixX2<scalar_T> layer_xy = trunk_layer_xyz(layer_point_indices, {0, 1}).matrix();
    KDTree2<scalar_T> *kd_tree_2d = new KDTree2<scalar_T>(2, std::cref(layer_xy), 10 /* max leaf size */);

    for (int64_t label = 0; label < num_labels; ++label) {
      std::vector<int64_t> current_batch_indices;

      int64_t flat_idx = label * num_layers + layer;

      bool is_circle_or_ellipse = preliminary_layer_circles_or_ellipses(flat_idx, 2) != -1;
      if (!is_circle_or_ellipse) {
        continue;
      }
      bool is_circle = preliminary_layer_circles_or_ellipses(flat_idx, 3) == -1;
      if (is_circle) {
        ArrayX<scalar_T> circle_center = preliminary_layer_circles_or_ellipses(flat_idx, {0, 1});
        scalar_T circle_radius = preliminary_layer_circles_or_ellipses(flat_idx, 2);
        scalar_T buffer_width;
        if (circle_radius * 2 <= trunk_search_circle_fitting_switch_buffer_threshold) {
          buffer_width = trunk_search_circle_fitting_small_buffer_width;
        } else {
          buffer_width = trunk_search_circle_fitting_large_buffer_width;
        }

        std::vector<nanoflann::ResultItem<int64_t, scalar_T>> search_result;

        const size_t num_neighbors =
            kd_tree_2d->index_->radiusSearch(circle_center.data(), circle_radius + buffer_width, search_result);

        auto min_radius_squared = circle_radius - buffer_width;
        min_radius_squared = min_radius_squared * min_radius_squared;

        for (int64_t i = 0; i < num_neighbors; ++i) {
          auto idx = search_result[i].first;
          auto dist = search_result[i].second;

          if (dist >= min_radius_squared) {
            current_batch_indices.push_back(layer_point_indices[idx]);
          }
        }
      } else {
        auto ellipse = preliminary_layer_circles_or_ellipses(flat_idx, Eigen::all);
        ArrayX2<scalar_T> ellipse_center = ellipse.leftCols(2);
        scalar_T ellipse_diameter = ellipse(2) + ellipse(3);
        scalar_T buffer_width;

        if (ellipse_diameter <= trunk_search_circle_fitting_switch_buffer_threshold) {
          buffer_width = trunk_search_circle_fitting_small_buffer_width;
        } else {
          buffer_width = trunk_search_circle_fitting_large_buffer_width;
        }

        std::vector<nanoflann::ResultItem<int64_t, scalar_T>> search_result;

        const size_t num_neighbors =
            kd_tree_2d->index_->radiusSearch(ellipse_center.data(), ellipse(2) + buffer_width, search_result);

        std::vector<int64_t> neighbor_indices;

        for (int64_t i = 0; i < num_neighbors; ++i) {
          neighbor_indices.push_back(layer_point_indices[search_result[i].first]);
        }

        ArrayX<scalar_T> outer_ellipse = ellipse;
        outer_ellipse({2, 3}) += buffer_width;
        ArrayX<scalar_T> inner_ellipse = ellipse;
        inner_ellipse({2, 3}) -= buffer_width;
        ArrayX2<scalar_T> neighbor_points = trunk_layer_xyz(neighbor_indices, {0, 1});
        ArrayXb is_in_outer_ellipse = points_in_ellipse<scalar_T>(neighbor_points, outer_ellipse);
        ArrayXb is_in_inner_ellipse = points_in_ellipse<scalar_T>(neighbor_points, inner_ellipse);

        for (int64_t i = 0; i < num_neighbors; ++i) {
          if (is_in_outer_ellipse(i) && !is_in_inner_ellipse(i)) {
            current_batch_indices.push_back(neighbor_indices[i]);
          }
        }
      }

      if (current_batch_indices.size() >= trunk_search_circle_fitting_min_points) {
        batch_indices[flat_idx] = current_batch_indices;
        batch_lengths[flat_idx] = current_batch_indices.size();
      }
    }
  }

  ArrayXl selected_indices(batch_lengths.sum());
  int64_t start_idx = 0;
  for (int64_t i = 0; i < batch_indices.size(); ++i) {
    selected_indices(Eigen::seqN(start_idx, batch_lengths(i))) =
        Eigen::Map<ArrayXl>(batch_indices[i].data(), batch_indices[i].size());
    start_idx += batch_lengths(i);
  }

  ArrayX<scalar_T> layer_heights = layer_bounds.rowwise().mean();

  return std::make_tuple(trunk_layer_xyz(selected_indices, {0, 1}), batch_lengths, layer_heights);
}

template <typename scalar_T>
std::tuple<ArrayXl, std::vector<int64_t>> collect_region_growing_seeds(
    RefArrayX3<scalar_T> xyz,
    RefArrayX<scalar_T> distance_to_dtm,
    RefArrayX2<scalar_T> tree_positions,
    RefArrayX<scalar_T> trunk_diameters,
    scalar_T region_growing_seed_layer_height,
    scalar_T region_growing_seed_radius_factor,
    int num_workers = 1) {
  if (xyz.rows() != distance_to_dtm.rows()) {
    throw std::invalid_argument("xyz and distance_to_dtm must have the same length.");
  }
  if (tree_positions.rows() != trunk_diameters.rows()) {
    throw std::invalid_argument("tree_positions and trunk_diameters must have the same length.");
  }

  auto num_trees = tree_positions.rows();
  auto num_points = xyz.rows();

  ArrayXl instance_ids = ArrayXl::Constant(num_points, -1);

  MatrixX2<scalar_T> xy_mat = xyz.leftCols(2).matrix();
  KDTree2<scalar_T> *kd_tree_2d = new KDTree2<scalar_T>(2, std::cref(xy_mat), 10 /* max leaf size */);

  std::vector<int64_t> seed_indices = {};

  ArrayX<scalar_T> search_radii = trunk_diameters / 2 * region_growing_seed_radius_factor;

  for (int64_t tree_id = 0; tree_id < num_trees; ++tree_id) {
    ArrayX<scalar_T> tree_position = tree_positions.row(tree_id);

    std::vector<nanoflann::ResultItem<int64_t, scalar_T>> search_result;

    const size_t num_neighbors =
        kd_tree_2d->index_->radiusSearch(tree_position.data(), search_radii(tree_id), search_result);

    bool found_seed_points = false;

    for (size_t i = 0; i < num_neighbors; i++) {
      auto idx = search_result[i].first;
      scalar_T height_above_ground = distance_to_dtm(idx);

      if ((height_above_ground >= 1.3 - region_growing_seed_layer_height / 2) &&
          (height_above_ground <= 1.3 + region_growing_seed_layer_height / 2)) {
        found_seed_points = true;
        instance_ids[idx] = tree_id;
        seed_indices.push_back(idx);
      }
    }

    if (!found_seed_points) {
      std::cout << "No seed points were found for tree " << tree_id << std::endl;
    }
  }

  delete kd_tree_2d;

  return std::make_tuple(instance_ids, seed_indices);
}

template <typename scalar_T>
ArrayXl segment_tree_crowns(
    RefArrayX3<scalar_T> xyz,
    RefArrayX<scalar_T> distance_to_dtm,
    RefArrayXb is_tree,
    RefArrayX2<scalar_T> tree_positions,
    RefArrayX<scalar_T> trunk_diameters,
    scalar_T region_growing_voxel_size,
    scalar_T region_growing_z_scale,
    scalar_T region_growing_seed_layer_height,
    scalar_T region_growing_seed_radius_factor,
    scalar_T region_growing_min_total_assignment_ratio,
    scalar_T region_growing_min_tree_assignment_ratio,
    scalar_T region_growing_max_search_radius,
    int64_t region_growing_decrease_search_radius_after_num_iter,
    int64_t region_growing_max_iterations,
    scalar_T region_growing_cum_search_dist_include_terrain,
    int num_workers) {
  if (num_workers <= 0) {
    num_workers = omp_get_max_threads();
  }

  if (xyz.rows() != is_tree.rows()) {
    throw std::invalid_argument("xyz and is_tree must have the same length.");
  }

  auto num_trees = tree_positions.rows();
  auto num_points = xyz.rows();

  std::tuple<ArrayXl, std::vector<int64_t>> region_growing_seeds = collect_region_growing_seeds<scalar_T>(
      xyz, distance_to_dtm, tree_positions, trunk_diameters, region_growing_seed_layer_height,
      region_growing_seed_radius_factor);
  ArrayXl instance_ids = std::get<0>(region_growing_seeds);
  std::vector<int64_t> seed_indices = std::get<1>(region_growing_seeds);

  xyz.col(2) = xyz.col(2) / region_growing_z_scale;

  scalar_T search_radius = region_growing_voxel_size;
  scalar_T search_radius_squared = search_radius * search_radius;
  int64_t iterations_without_radius_increase = 0;
  scalar_T cumulative_search_dist = 0;

  for (int64_t i = 0; i < region_growing_max_iterations; i++) {
    if (search_radius > region_growing_max_search_radius || seed_indices.size() == 0) {
      break;
    }

    std::vector<int64_t> unassigned_indices = {};

    if (cumulative_search_dist <= region_growing_cum_search_dist_include_terrain) {
      for (int64_t j = 0; j < num_points; ++j) {
        if (instance_ids[j] == -1) {
          unassigned_indices.push_back(j);
        }
      }
    } else {
      for (int64_t j = 0; j < num_points; ++j) {
        if (instance_ids[j] == -1 && is_tree[j]) {
          unassigned_indices.push_back(j);
        }
      }
    }
    if (unassigned_indices.size() == 0) {
      break;
    }
    ArrayX3<scalar_T> unassigned_xyz = xyz(unassigned_indices, Eigen::all);

    std::cout << "Iteration " << i << ", " << unassigned_indices.size()
              << " unassigned points remaining, search radius: " << search_radius << ", seeds: " << seed_indices.size()
              << "." << std::endl;

    MatrixX3<scalar_T> seed_xyz = xyz(seed_indices, Eigen::all).matrix();
    KDTree3<scalar_T> kd_tree_seeds_3d(3, std::cref(seed_xyz), 10 /* max leaf size */);

    ArrayXb becomes_new_seed = ArrayXb::Constant(unassigned_indices.size(), 0);
    ArrayXb tree_was_grown = ArrayXb::Constant(num_trees, 0);

#pragma omp parallel for num_threads(num_workers)
    for (int64_t j = 0; j < unassigned_indices.size(); ++j) {
      std::vector<int64_t> knn_index(1);
      std::vector<scalar_T> knn_dist(1);

      ArrayX3<scalar_T> query_xyz = unassigned_xyz.row(j);

      auto num_results = kd_tree_seeds_3d.index_->knnSearch(query_xyz.data(), 1, &knn_index[0], &knn_dist[0]);

      assert(num_results == 1);

      if (knn_dist[0] > search_radius_squared) {
        continue;
      }

      auto instance_id = instance_ids(seed_indices[knn_index[0]]);
      assert(instance_id != -1);
      instance_ids[unassigned_indices[j]] = instance_id;
      becomes_new_seed[j] = true;
      tree_was_grown[instance_id] = true;
    }

    scalar_T newly_assigned_points_ratio = (scalar_T)becomes_new_seed.count() / (scalar_T)unassigned_indices.size();
    scalar_T tree_assignment_ratio = (scalar_T)tree_was_grown.count() / (scalar_T)num_trees;

    std::cout << "newly_assigned_points_ratio: " << newly_assigned_points_ratio
              << ", tree_assignment_ratio: " << tree_assignment_ratio << "." << std::endl;

    if (newly_assigned_points_ratio < region_growing_min_total_assignment_ratio ||
        tree_assignment_ratio < region_growing_min_tree_assignment_ratio) {
      search_radius += region_growing_voxel_size;

      seed_indices.clear();
      for (int64_t j = 0; j < num_points; ++j) {
        if (instance_ids[j] != -1) {
          seed_indices.push_back(j);
        }
      }

      iterations_without_radius_increase = 0;
    } else {
      seed_indices.clear();
      for (int64_t j = 0; j < unassigned_indices.size(); ++j) {
        if (becomes_new_seed[j]) {
          seed_indices.push_back(unassigned_indices[j]);
        }
      }

      iterations_without_radius_increase += 1;
    }

    if (iterations_without_radius_increase == region_growing_decrease_search_radius_after_num_iter) {
      search_radius -= region_growing_voxel_size;
      iterations_without_radius_increase = 0;
    }

    search_radius_squared = search_radius * search_radius;
    cumulative_search_dist += search_radius;
  }

  return instance_ids;
}

}  // namespace PointTree

#endif  // TREE_X_ALGORITHM_H
