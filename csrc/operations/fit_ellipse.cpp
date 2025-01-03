#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace Eigen;
using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;
using ArrayX5d = Eigen::Array<double, Eigen::Dynamic, 5>;
using ArrayXl = Eigen::Array<int64_t, Eigen::Dynamic, 1>;

double stddev(ArrayX2d x) {
  double mean = x.mean();
  double variance = (x - mean).square().mean();
  double stddev = std::sqrt(variance);

  return stddev;
}

}  // namespace

ArrayX5d fit_ellipse(ArrayX2d xy, ArrayXl batch_lengths, int num_workers = 1) {
  /*
  This C++ implementation is based on the Python implementation from the scikit-image package:

  https://github.com/scikit-image/scikit-image/blob/efe339b09ee7d9d8eb163d6750005ab0bef703b8/skimage/measure/fit.py

  The scikit-image implementation is based on the Python implementation by Ben Hammel and Nick Sullivan-Molina:

  https://github.com/bdhammel/least-squares-ellipse-fitting/

  The implementation references the following resources:

  [1] Halir, Radim, and Jan Flusser. "Numerically Stable Direct Least Squares Fitting of Ellipses." Proc. 6th
  International Conference in Central Europe on Computer Graphics and Visualization. WSCG. Vol. 98. Plzen-Bory:
  Citeseer, 1998.
  [2] Weisstein, Eric W. "Ellipse." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/Ellipse.html
  */

  if (num_workers != -1) {
    omp_set_num_threads(num_workers);
  }

  constexpr double PI = 3.14159265358979311600;

  if (xy.rows() != batch_lengths.sum()) {
    throw std::invalid_argument("The number of points must be equal to the sum of batch_lengths.");
  }

  auto num_batches = batch_lengths.size();

  ArrayXl batch_starts(num_batches);

  int64_t batch_start = 0;
  for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    batch_starts(batch_idx) = batch_start;
    batch_start += batch_lengths(batch_idx);
  }

  ArrayX5d ellipse_params(num_batches, 5);

#pragma omp parallel for num_threads(num_workers)
  for (int64_t i = 0; i < num_batches; ++i) {
    ArrayX2d current_xy = xy(seqN(batch_starts(i), batch_lengths(i)), Eigen::all);

    if (current_xy.rows() == 0) {
      ellipse_params.row(i) = -1;
      continue;
    }

    // normalize value range to avoid misfitting due to numeric errors if
    // the relative distanceses are small compared to absolute distances
    ArrayXd origin = current_xy.colwise().mean();

    current_xy = current_xy.rowwise() - origin.transpose();

    double scale = stddev(current_xy);

    current_xy = current_xy / scale;

    // quadratic part of design matrix [eqn. 15] from [1]
    MatrixX3d D1(current_xy.rows(), 3);
    D1(Eigen::all, 0) = current_xy.col(0) * current_xy.col(0);
    D1(Eigen::all, 1) = current_xy.col(0) * current_xy.col(1);
    D1(Eigen::all, 2) = current_xy.col(1) * current_xy.col(1);

    // linear part of design matrix [eqn. 16] from [1]
    MatrixX3d D2(current_xy.rows(), 3);
    D2(Eigen::all, 0) = current_xy.col(0);
    D2(Eigen::all, 1) = current_xy.col(1);
    D2(Eigen::all, 2) = ArrayXd::Constant(current_xy.rows(), 1);

    // forming scatter matrix [eqn. 17] from [1]
    MatrixX3d S1 = D1.matrix().transpose() * D1.matrix();
    MatrixX3d S2 = D1.matrix().transpose() * D2.matrix();
    MatrixX3d S3 = D2.matrix().transpose() * D2.matrix();

    // constraint matrix [eqn. 18]
    MatrixX3d C1(3, 3);
    C1 << 0.0, 0.0, 2.0, 0.0, -1.0, 0.0, 2.0, 0.0, 0.0;

    MatrixX3d M;
    if (C1.matrix().determinant() != 0 && S3.matrix().determinant() != 0) {
      // Reduced scatter matrix [eqn. 29]
      M = C1.inverse() * (S1 - S2 * S3.inverse() * S2.transpose());
    } else {
      ellipse_params.row(i) = -1;
      continue;
    }

    Eigen::EigenSolver<MatrixX3d> eigen_solver(M);

    // M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors
    // from this equation [eqn. 28]
    ArrayX3d eigen_vectors = eigen_solver.eigenvectors().real();
    ArrayXd eigen_values = eigen_solver.eigenvalues().real();

    if ((eigen_solver.eigenvectors().imag().array() != 0).sum()) {
      ellipse_params.row(i) = -1;
      continue;
    }

    // eigenvector must meet constraint 4ac - b^2 to be valid.
    ArrayXb cond = (4 * eigen_vectors.row(0) * eigen_vectors.row(2) - eigen_vectors.row(1) * eigen_vectors.row(1)) > 0;

    if (cond.sum() != 1) {
      ellipse_params.row(i) = -1;
      continue;
    }

    Eigen::Index a1_index;
    cond.maxCoeff(&a1_index);
    ArrayXd a1 = eigen_vectors.col(a1_index);

    double a = a1(0);
    double b = a1(1);
    double c = a1(2);

    // |d e g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
    ArrayXd a2 = -S3.inverse() * S2.transpose() * a1.matrix();

    double d = a2(0);
    double e = a2(1);
    double f = a2(2);

    // eigenvectors are the coefficients of an ellipse in general form
    // a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*e*y + f = 0 (eqn. 15) from [2]
    b /= 2.0;
    d /= 2.0;
    e /= 2.0;

    // finding center of ellipse [eqn.19 and 20] from [2]
    double center_x = (c * d - b * e) / (b * b - a * c);
    double center_y = (a * e - b * d) / (b * b - a * c);

    // find the semi-axes lengths [eqn. 21 and 22] from [2]
    double numerator = a * e * e + c * d * d + f * b * b - 2 * b * d * e - a * c * f;
    double term = sqrt(pow((a - c), 2) + 4 * b * b);
    double denominator1 = (b * b - a * c) * (term - (a + c));
    double denominator2 = (b * b - a * c) * (-term - (a + c));
    double radius_major = sqrt(2 * numerator / denominator1);
    double radius_minor = sqrt(2 * numerator / denominator2);

    // angle of counterclockwise rotation of major-axis of ellipse to x-axis [eqn. 23] from [2].
    double phi = 0.5 * atan((2.0 * b) / (a - c));
    if (a > c) {
      phi += 0.5 * PI;
    }

    // stabilize parameters: sometimes small fluctuations in data can cause height and width to swap
    if (radius_major < radius_minor) {
      double radius_major_old = radius_major;
      radius_major = radius_minor;
      radius_minor = radius_major_old;
      phi += PI / 2;
    }

    phi = fmod(phi, PI);

    ellipse_params.row(i) << center_x, center_y, radius_major, radius_minor, phi;

    // revert normalization
    ellipse_params(i, {0, 1, 2, 3}) *= scale;
    ellipse_params(i, {0, 1}) += origin;
  }

  return ellipse_params;
}
