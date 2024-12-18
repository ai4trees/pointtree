#include <omp.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <nanoflann.hpp>
#include <vector>

namespace {

using ArrayXd = Eigen::Array<double, Eigen::Dynamic, 1>;
using ArrayX5d = Eigen::Array<double, Eigen::Dynamic, 5>;
using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;
using ArrayXl = Eigen::Array<int64_t, Eigen::Dynamic, 1>;
using MatrixX2d = Eigen::Matrix<double, Eigen::Dynamic, 2>;
using MatrixX3d = Eigen::Matrix<double, Eigen::Dynamic, 3>;
using KDTree2d =
    nanoflann::KDTreeEigenMatrixAdaptor<MatrixX2d, 2,
                                        nanoflann::metric_L2_Simple>;
using KDTree3d =
    nanoflann::KDTreeEigenMatrixAdaptor<MatrixX3d, 3,
                                        nanoflann::metric_L2_Simple>;

using namespace Eigen;

ArrayXl
segment_tree_crowns(ArrayX3d xyz, ArrayXd distance_to_dtm, ArrayXb is_tree,
                    ArrayX2d tree_positions, ArrayXd trunk_diameters,
                    double region_growing_voxel_size,
                    double region_growing_z_scale,
                    double region_growing_seed_layer_height,
                    double region_growing_seed_radius_factor,
                    double region_growing_min_total_assignment_ratio,
                    double region_growing_min_tree_assignment_ratio,
                    double region_growing_max_search_radius,
                    int region_growing_decrease_search_radius_after_num_iter,
                    int region_growing_max_iterations,
                    double region_growing_cum_search_dist_include_terrain) {
  auto num_trees = tree_positions.rows();
  auto num_points = xyz.rows();

  ArrayXl instance_ids = ArrayXl::Constant(num_points, -1);

  MatrixX2d xy_mat = xyz.leftCols(2).matrix();
  KDTree2d *kd_tree_2d =
      new KDTree2d(2, std::cref(xy_mat), 10 /* max leaf size */);

  std::vector<int64_t> seed_indices = {};

  ArrayXd search_radii =
      trunk_diameters / 2 * region_growing_seed_radius_factor;

  for (int tree_id = 0; tree_id < num_trees; ++tree_id) {
    ArrayXd tree_position = tree_positions.row(tree_id);

    std::vector<nanoflann::ResultItem<int64_t, double>> search_result;

    const size_t num_neighbors = kd_tree_2d->index_->radiusSearch(
        tree_position.data(), search_radii(tree_id), search_result);

    bool found_seed_points = false;

    for (size_t i = 0; i < num_neighbors; i++) {
      auto idx = search_result[i].first;
      double height_above_ground = distance_to_dtm(idx);

      if ((height_above_ground >= 1.3 - region_growing_seed_layer_height / 2) &&
          (height_above_ground <= 1.3 + region_growing_seed_layer_height / 2)) {
        found_seed_points = true;
        instance_ids[idx] = tree_id;
        seed_indices.push_back(idx);
      }
    }

    if (!found_seed_points) {
      std::cout << "No seed points were found for tree " << tree_id
                << std::endl;
    }
  }

  delete kd_tree_2d;

  xyz.col(2) = xyz.col(2) / region_growing_z_scale;

  double search_radius = region_growing_voxel_size;
  double search_radius_squared = search_radius * search_radius;
  int iterations_without_radius_increase = 0;
  double cumulative_search_dist = 0;

  for (int i = 0; i < region_growing_max_iterations; i++) {
    if (search_radius > region_growing_max_search_radius ||
        seed_indices.size() == 0) {
      break;
    }

    std::vector<int64_t> unassigned_indices = {};

    if (cumulative_search_dist <=
        region_growing_cum_search_dist_include_terrain) {
      for (int j = 0; j < num_points; ++j) {
        if (instance_ids[j] == -1) {
          unassigned_indices.push_back(j);
        }
      }
    } else {
      for (int j = 0; j < num_points; ++j) {
        if (instance_ids[j] == -1 && is_tree[j]) {
          unassigned_indices.push_back(j);
        }
      }
    }
    if (unassigned_indices.size() == 0) {
      break;
    }
    ArrayX3d unassigned_xyz = xyz(unassigned_indices, Eigen::all);

    std::cout << "Iteration " << i << ", " << unassigned_indices.size()
              << " unassigned points remaining, search radius: "
              << search_radius << ", seeds: " << seed_indices.size() << "."
              << std::endl;

    MatrixX3d seed_xyz = xyz(seed_indices, Eigen::all).matrix();
    KDTree3d kd_tree_seeds_3d(3, std::cref(seed_xyz), 10 /* max leaf size */);

    ArrayXb becomes_new_seed = ArrayXb::Constant(unassigned_indices.size(), 0);
    ArrayXb tree_was_grown = ArrayXb::Constant(num_trees, 0);

#pragma omp parallel for
    for (int j = 0; j < unassigned_indices.size(); ++j) {
      std::vector<int64_t> knn_index(1);
      std::vector<double> knn_dist(1);

      ArrayX3d query_xyz = unassigned_xyz.row(j);

      auto num_results = kd_tree_seeds_3d.index_->knnSearch(
          query_xyz.data(), 1, &knn_index[0], &knn_dist[0]);

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

    double newly_assigned_points_ratio =
        (double)becomes_new_seed.count() / (double)unassigned_indices.size();
    double tree_assignment_ratio =
        (double)tree_was_grown.count() / (double)num_trees;

    std::cout << "newly_assigned_points_ratio: " << newly_assigned_points_ratio
              << ", tree_assignment_ratio: " << tree_assignment_ratio << "."
              << std::endl;

    if (newly_assigned_points_ratio <
            region_growing_min_total_assignment_ratio ||
        tree_assignment_ratio < region_growing_min_tree_assignment_ratio) {
      search_radius += region_growing_voxel_size;

      seed_indices.clear();
      for (int j = 0; j < num_points; ++j) {
        if (instance_ids[j] != -1) {
          seed_indices.push_back(j);
        }
      }

      iterations_without_radius_increase = 0;
    } else {
      seed_indices.clear();
      for (int j = 0; j < unassigned_indices.size(); ++j) {
        if (becomes_new_seed[j]) {
          seed_indices.push_back(unassigned_indices[j]);
        }
      }

      iterations_without_radius_increase += 1;
    }

    if (iterations_without_radius_increase ==
        region_growing_decrease_search_radius_after_num_iter) {
      search_radius -= region_growing_voxel_size;
      iterations_without_radius_increase = 0;
    }

    search_radius_squared = search_radius * search_radius;
    cumulative_search_dist += search_radius;
  }

  return instance_ids;
}

} // namespace

PYBIND11_MODULE(_tree_x_algorithm_cpp, m) {
  m.doc() = R"pbdoc(
    C++ extension module implementing selected steps of the TreeXAlgorithm.
  )pbdoc";

  m.def("segment_tree_crowns", &segment_tree_crowns,
        pybind11::return_value_policy::reference_internal, R"pbdoc(
    C++ implementation of the region-growing method for tree crown segmentation. For more details, see the documentation of the Python wrapper method
    :code:`TreeXAlgorithm.segment_tree_crowns()`.
  )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = (VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
