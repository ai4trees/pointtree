#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../type_aliases.h"

#ifndef DISTANCE_TO_DTM_H
#define DISTANCE_TO_DTM_H

namespace PointTree {

template <typename scalar_T>
ArrayX<scalar_T> distance_to_dtm(
    RefArrayX3<scalar_T> xyz,
    RefArrayXX<scalar_T> dtm,
    RefArray2<scalar_T> dtm_offset,
    scalar_T dtm_resolution,
    bool allow_outside_points = true) {
  ArrayX2<scalar_T> grid_positions = ((xyz(Eigen::all, {0, 1}).rowwise() - dtm_offset) / dtm_resolution);

  if (allow_outside_points) {
    // clamp grid positions to be within the DTM boundaries
    grid_positions = grid_positions.max(0);
    grid_positions.col(0) = grid_positions.col(0).min(dtm.cols() - 1);
    grid_positions.col(1) = grid_positions.col(1).min(dtm.rows() - 1);
  }

  ArrayX2l grid_indices = grid_positions.floor().cast<int64_t>();

  if (!allow_outside_points && ((grid_positions < 0).any() || (grid_positions.col(0) >= dtm.cols()).any() ||
                                (grid_positions.col(1) >= dtm.rows()).any())) {
    throw std::invalid_argument("The DTM does not completely cover the point cloud to be normalized.");
  }

  ArrayX2<scalar_T> grid_fractions = grid_positions - grid_positions.floor();

  Eigen::Map<ArrayX<scalar_T>> dtm_flat(dtm.data(), dtm.size());

  auto height_1 = dtm_flat(grid_indices(Eigen::all, 0) * dtm.rows() + grid_indices(Eigen::all, 1));
  auto height_2 =
      dtm_flat((grid_indices(Eigen::all, 0) + 1).min(dtm.cols() - 1) * dtm.rows() + grid_indices(Eigen::all, 1));
  auto height_3 =
      dtm_flat(grid_indices(Eigen::all, 0) * dtm.rows() + (grid_indices(Eigen::all, 1) + 1).min(dtm.rows() - 1));
  auto height_4 = dtm_flat(
      (grid_indices(Eigen::all, 0) + 1).min(dtm.cols() - 1) * dtm.rows() +
      (grid_indices(Eigen::all, 1) + 1).min(dtm.rows() - 1));

  auto interp_height_1 = height_1 * (1 - grid_fractions(Eigen::all, 0)) + height_2 * (grid_fractions(Eigen::all, 0));
  auto interp_height_2 = height_3 * (1 - grid_fractions(Eigen::all, 0)) + height_4 * (grid_fractions(Eigen::all, 0));
  ArrayX<scalar_T> terrain_height =
      interp_height_1 * (1 - grid_fractions(Eigen::all, 1)) + interp_height_2 * (grid_fractions(Eigen::all, 1));

  return xyz(Eigen::all, 2) - terrain_height;
}

}  // namespace PointTree

#endif  // DISTANCE_TO_DTM_H
