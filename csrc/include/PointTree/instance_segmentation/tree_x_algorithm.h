#include <omp.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <nanoflann.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../operations/distance_to_dtm.h"
#include "../operations/points_in_ellipse.h"
#include "../type_aliases.h"

#ifndef TREE_X_ALGORITHM_H
#define TREE_X_ALGORITHM_H

namespace PointTree {

template <typename scalar_T>
ArrayX2<scalar_T> compute_layer_bounds(
    scalar_T start_z, int64_t num_layers, scalar_T layer_height, scalar_T layer_overlap) {
  ArrayX2<scalar_T> layer_bounds;
  layer_bounds.resize(num_layers, 2);

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
std::tuple<ArrayX2<scalar_T>, ArrayXl, ArrayX<scalar_T>, ArrayX<scalar_T>> collect_inputs_trunk_layers_fitting(
    RefArrayX3<scalar_T> trunk_layer_xyz,
    RefArrayXl cluster_labels,
    RefArrayXl unique_cluster_labels,
    RefArrayXX<scalar_T> dtm,
    RefArray2<scalar_T> dtm_offset,
    scalar_T dtm_resolution,
    scalar_T trunk_search_circle_fitting_layer_start,
    int64_t trunk_search_circle_fitting_num_layers,
    scalar_T trunk_search_circle_fitting_layer_height,
    scalar_T trunk_search_circle_fitting_layer_overlap,
    int64_t trunk_search_circle_fitting_min_points = 0,
    int num_workers = -1) {
  if (num_workers <= 0) {
    num_workers = omp_get_max_threads();
  }

  if (trunk_layer_xyz.rows() != cluster_labels.rows()) {
    throw std::invalid_argument("The length of trunk_layer_xyz and cluster_labels must be equal.");
  }

  auto num_labels = unique_cluster_labels.rows();
  auto num_layers = trunk_search_circle_fitting_num_layers;
  std::vector<std::vector<int64_t>> batch_point_indices(num_labels * num_layers);
  ArrayXl batch_lengths = ArrayXl::Constant(num_labels * num_layers, 0);

  ArrayX2<scalar_T> layer_bounds = compute_layer_bounds<scalar_T>(
      trunk_search_circle_fitting_layer_start, trunk_search_circle_fitting_num_layers,
      trunk_search_circle_fitting_layer_height, trunk_search_circle_fitting_layer_overlap);

  std::vector<std::vector<int64_t>> current_cluster_point_indices(num_labels);
  ArrayX<scalar_T> terrain_heights = ArrayX<scalar_T>::Constant(num_labels, 0);

#pragma omp parallel for num_threads(num_workers)
  for (int64_t idx = 0; idx < num_labels; ++idx) {
    for (int64_t i = 0; i < trunk_layer_xyz.rows(); ++i) {
      if (cluster_labels(i) == unique_cluster_labels(idx)) {
        current_cluster_point_indices[idx].push_back(i);
      }
    }

    // compute terrain height at cluster center position
    ArrayX3<scalar_T> cluster_center_xyz = ArrayX3<scalar_T>::Constant(1, 3, 0);
    cluster_center_xyz(0, {0, 1}) = trunk_layer_xyz(current_cluster_point_indices[idx], {0, 1}).colwise().mean();

    terrain_heights(idx) = -1 * distance_to_dtm<scalar_T>(cluster_center_xyz, dtm, dtm_offset, dtm_resolution, true)(0);
  }

#pragma omp parallel for num_threads(num_workers)
  for (int64_t idx = 0; idx < num_labels * num_layers; ++idx) {
    int64_t label = idx / num_layers;
    int64_t layer = idx % num_layers;
    std::vector<int64_t> current_cluster_layer_point_indices;

    for (int64_t i = 0; i < current_cluster_point_indices[label].size(); ++i) {
      auto j = current_cluster_point_indices[label][i];
      auto dist_to_dtm = trunk_layer_xyz(j, 2) - terrain_heights(label);
      if (dist_to_dtm >= layer_bounds(layer, 0) && dist_to_dtm <= layer_bounds(layer, 1)) {
        current_cluster_layer_point_indices.push_back(j);
      }
    }
    if (current_cluster_layer_point_indices.size() >= trunk_search_circle_fitting_min_points) {
      batch_point_indices[idx] = current_cluster_layer_point_indices;
      batch_lengths[idx] = current_cluster_layer_point_indices.size();
    }
  }

  ArrayXl selected_indices(batch_lengths.sum());
  int64_t start_idx = 0;
  for (int64_t i = 0; i < batch_point_indices.size(); ++i) {
    selected_indices(Eigen::seqN(start_idx, batch_lengths(i))) =
        Eigen::Map<ArrayXl>(batch_point_indices[i].data(), batch_point_indices[i].size());
    start_idx += batch_lengths(i);
  }

  ArrayX<scalar_T> layer_heights = layer_bounds.rowwise().mean();

  return std::make_tuple(trunk_layer_xyz(selected_indices, {0, 1}), batch_lengths, terrain_heights, layer_heights);
}

template <typename scalar_T>
std::tuple<ArrayX2<scalar_T>, ArrayXl> collect_inputs_trunk_layers_refined_fitting(
    RefArrayX3<scalar_T> trunk_layer_xyz,
    RefArrayX3<scalar_T> preliminary_layer_circles,
    RefArrayX5<scalar_T> preliminary_layer_ellipses,
    RefArrayX<scalar_T> terrain_heights,
    scalar_T trunk_search_circle_fitting_layer_start,
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
  auto num_labels = preliminary_layer_circles.rows() / num_layers;
  std::vector<std::vector<int64_t>> batch_point_indices(num_labels * num_layers);
  ArrayXl batch_lengths = ArrayXl::Constant(preliminary_layer_circles.rows(), 0);

  ArrayX2<scalar_T> layer_bounds = compute_layer_bounds<scalar_T>(
      trunk_search_circle_fitting_layer_start, trunk_search_circle_fitting_num_layers,
      trunk_search_circle_fitting_layer_height, trunk_search_circle_fitting_layer_overlap);

  MatrixX2<scalar_T> trunk_layer_xy = trunk_layer_xyz(Eigen::all, {0, 1}).matrix();
  KDTree2<scalar_T> *kd_tree_2d = new KDTree2<scalar_T>(2, std::cref(trunk_layer_xy), 10 /* max leaf size */);

#pragma omp parallel for num_threads(num_workers)
  for (int64_t label = 0; label < num_labels; ++label) {
    for (int64_t layer = 0; layer < num_layers; ++layer) {
      std::vector<int64_t> current_batch_point_indices;

      int64_t flat_idx = label * num_layers + layer;

      bool is_circle = preliminary_layer_circles(flat_idx, 2) != -1;
      bool is_ellipse = preliminary_layer_ellipses(flat_idx, 2) != -1;

      if (!is_circle && !is_ellipse) {
        continue;
      }
      if (is_circle) {
        ArrayX<scalar_T> circle_center = preliminary_layer_circles(flat_idx, {0, 1});
        scalar_T circle_radius = preliminary_layer_circles(flat_idx, 2);
        scalar_T buffer_width;
        if (circle_radius * 2 <= trunk_search_circle_fitting_switch_buffer_threshold) {
          buffer_width = trunk_search_circle_fitting_small_buffer_width;
        } else {
          buffer_width = trunk_search_circle_fitting_large_buffer_width;
        }

        std::vector<nanoflann::ResultItem<int64_t, scalar_T>> search_result;

        auto min_radius_squared = circle_radius - buffer_width;
        min_radius_squared = min_radius_squared * min_radius_squared;
        // the search radius needs to be squared since the KDTree uses the squared L2 norm
        auto max_radius_squared = circle_radius + buffer_width;
        max_radius_squared = max_radius_squared * max_radius_squared;

        const size_t num_neighbors =
            kd_tree_2d->index_->radiusSearch(circle_center.data(), max_radius_squared, search_result);

        for (int64_t i = 0; i < num_neighbors; ++i) {
          auto idx = search_result[i].first;
          auto dist = search_result[i].second;
          auto dist_to_dtm = trunk_layer_xyz(idx, 2) - terrain_heights(label);

          if (dist_to_dtm >= layer_bounds(layer, 0) && dist_to_dtm <= layer_bounds(layer, 1) &&
              dist >= min_radius_squared) {
            current_batch_point_indices.push_back(idx);
          }
        }
      } else {
        auto ellipse = preliminary_layer_ellipses(flat_idx, Eigen::all);
        ArrayX2<scalar_T> ellipse_center = ellipse.leftCols(2);
        scalar_T ellipse_diameter = ellipse(2) + ellipse(3);
        scalar_T buffer_width;

        if (ellipse_diameter <= trunk_search_circle_fitting_switch_buffer_threshold) {
          buffer_width = trunk_search_circle_fitting_small_buffer_width;
        } else {
          buffer_width = trunk_search_circle_fitting_large_buffer_width;
        }

        std::vector<nanoflann::ResultItem<int64_t, scalar_T>> search_result;

        auto max_radius_squared = ellipse(2) + buffer_width;
        max_radius_squared = max_radius_squared * max_radius_squared;

        const size_t num_neighbors =
            kd_tree_2d->index_->radiusSearch(ellipse_center.data(), max_radius_squared, search_result);

        std::vector<int64_t> neighbor_indices;

        for (int64_t i = 0; i < num_neighbors; ++i) {
          auto idx = search_result[i].first;
          auto dist_to_dtm = trunk_layer_xyz(idx, 2) - terrain_heights(label);
          if (dist_to_dtm >= layer_bounds(layer, 0) && dist_to_dtm <= layer_bounds(layer, 1)) {
            neighbor_indices.push_back(idx);
          }
        }

        ArrayX<scalar_T> outer_ellipse = ellipse;
        outer_ellipse({2, 3}) += buffer_width;
        ArrayX<scalar_T> inner_ellipse = ellipse;
        inner_ellipse({2, 3}) -= buffer_width;
        ArrayX2<scalar_T> neighbor_xy = trunk_layer_xyz(neighbor_indices, {0, 1});
        ArrayXb is_in_outer_ellipse = points_in_ellipse<scalar_T>(neighbor_xy, outer_ellipse);
        ArrayXb is_in_inner_ellipse = points_in_ellipse<scalar_T>(neighbor_xy, inner_ellipse);

        for (int64_t i = 0; i < neighbor_indices.size(); ++i) {
          if (is_in_outer_ellipse(i) && !is_in_inner_ellipse(i)) {
            current_batch_point_indices.push_back(neighbor_indices[i]);
          }
        }
      }

      if (current_batch_point_indices.size() >= trunk_search_circle_fitting_min_points) {
        batch_point_indices[flat_idx] = current_batch_point_indices;
        batch_lengths[flat_idx] = current_batch_point_indices.size();
      }
    }
  }

  ArrayXl selected_indices(batch_lengths.sum());
  int64_t start_idx = 0;
  for (int64_t i = 0; i < batch_point_indices.size(); ++i) {
    selected_indices(Eigen::seqN(start_idx, batch_lengths(i))) =
        Eigen::Map<ArrayXl>(batch_point_indices[i].data(), batch_point_indices[i].size());
    start_idx += batch_lengths(i);
  }

  return std::make_tuple(trunk_layer_xyz(selected_indices, {0, 1}), batch_lengths);
}

template <typename scalar_T>
std::tuple<ArrayXl, std::vector<int64_t>> collect_region_growing_seeds(
    RefArrayX3<scalar_T> xyz,
    RefArrayX<scalar_T> distance_to_dtm,
    RefArrayX2<scalar_T> tree_positions,
    RefArrayX<scalar_T> trunk_diameters,
    RefArrayXl cluster_labels,
    scalar_T region_growing_seed_layer_height,
    scalar_T region_growing_seed_diameter_factor,
    scalar_T region_growing_seed_min_diameter,
    int num_workers = -1) {
  if (num_workers <= 0) {
    num_workers = omp_get_max_threads();
  }

  if (xyz.rows() != distance_to_dtm.rows()) {
    throw std::invalid_argument("xyz and distance_to_dtm must have the same length.");
  }
  if (tree_positions.rows() != trunk_diameters.rows()) {
    throw std::invalid_argument("tree_positions and trunk_diameters must have the same length.");
  }

  auto num_trees = tree_positions.rows();
  auto num_points = xyz.rows();

  ArrayXl instance_ids = ArrayXl::Constant(num_points, -1);

  std::vector<int64_t> seed_layer_indices;
  for (int64_t i = 0; i < xyz.rows(); ++i) {
    if ((distance_to_dtm(i) >= 1.3 - region_growing_seed_layer_height / 2) &&
        (distance_to_dtm(i) <= 1.3 + region_growing_seed_layer_height / 2) && cluster_labels(i) == -1) {
      seed_layer_indices.push_back(i);
    }
  }

  MatrixX2<scalar_T> seed_layer_xy = xyz(seed_layer_indices, {0, 1}).matrix();
  KDTree2<scalar_T> *kd_tree_2d = new KDTree2<scalar_T>(2, std::cref(seed_layer_xy), 10 /* max leaf size */);

  std::vector<int64_t> seed_indices = {};

  ArrayX<scalar_T> search_radii = trunk_diameters / 2 * region_growing_seed_diameter_factor;

#pragma omp parallel for num_threads(num_workers)
  for (int64_t tree_id = 0; tree_id < num_trees; ++tree_id) {
    ArrayX<scalar_T> tree_position = tree_positions.row(tree_id);

    std::vector<nanoflann::ResultItem<int64_t, scalar_T>> search_result;

    scalar_T search_radius = search_radii(tree_id) > region_growing_seed_min_diameter / 2
                                 ? search_radii(tree_id)
                                 : region_growing_seed_min_diameter / 2;
    // the search radius needs to be squared since the KDTree uses the squared L2 norm
    search_radius = search_radius * search_radius;
    const size_t num_neighbors = kd_tree_2d->index_->radiusSearch(tree_position.data(), search_radius, search_result);

    std::vector<int64_t> current_seed_indices(search_result.size());

    std::transform(
        search_result.begin(), search_result.end(), current_seed_indices.begin(),
        [seed_layer_indices](const nanoflann::ResultItem<int64_t, scalar_T> &x) {
          return seed_layer_indices[x.first];
        });

    for (int64_t i = 0; i < xyz.rows(); ++i) {
      if (cluster_labels(i) == tree_id) {
        current_seed_indices.push_back(i);
      }
    }

#pragma omp critical
    {
      seed_indices.reserve(seed_indices.size() + search_result.size());
      seed_indices.insert(seed_indices.end(), current_seed_indices.begin(), current_seed_indices.end());
      instance_ids(current_seed_indices) = tree_id;
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
    RefArrayXl cluster_labels,
    scalar_T region_growing_voxel_size,
    scalar_T region_growing_z_scale,
    scalar_T region_growing_seed_layer_height,
    scalar_T region_growing_seed_diameter_factor,
    scalar_T region_growing_seed_min_diameter,
    scalar_T region_growing_min_total_assignment_ratio,
    scalar_T region_growing_min_tree_assignment_ratio,
    scalar_T region_growing_max_search_radius,
    int64_t region_growing_decrease_search_radius_after_num_iter,
    int64_t region_growing_max_iterations,
    scalar_T region_growing_cum_search_dist_include_terrain,
    int num_workers = -1) {
  if (num_workers <= 0) {
    num_workers = omp_get_max_threads();
  }

  if (xyz.rows() != is_tree.rows()) {
    throw std::invalid_argument("xyz and is_tree must have the same length.");
  }

  auto num_trees = tree_positions.rows();
  auto num_points = xyz.rows();

  std::tuple<ArrayXl, std::vector<int64_t>> region_growing_seeds = collect_region_growing_seeds<scalar_T>(
      xyz, distance_to_dtm, tree_positions, trunk_diameters, cluster_labels, region_growing_seed_layer_height,
      region_growing_seed_diameter_factor, region_growing_seed_min_diameter);
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

      ArrayX3<scalar_T> query_xyz = xyz(unassigned_indices[j], Eigen::all);

      auto num_results = kd_tree_seeds_3d.index_->knnSearch(query_xyz.data(), 1, &knn_index[0], &knn_dist[0]);

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

  xyz.col(2) = xyz.col(2) * region_growing_z_scale;

  return instance_ids;
}

}  // namespace PointTree

#endif  // TREE_X_ALGORITHM_H
