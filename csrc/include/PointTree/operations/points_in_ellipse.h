#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "../type_aliases.h"

#ifndef POINTS_IN_ELLIPSE_H
#define POINTS_IN_ELLIPSE_H

namespace PointTree {

template <typename scalar_T>
ArrayXb points_in_ellipse(RefArrayX2<scalar_T> xy, RefArrayX<scalar_T> ellipse) {
  double center_x = ellipse(0);
  double center_y = ellipse(1);
  double radius_major = ellipse(2);
  double radius_minor = ellipse(3);
  double theta = ellipse(4);

  double cos_theta = cos(theta);
  double sin_theta = sin(theta);

  ArrayX<scalar_T> a = (cos_theta * (xy(Eigen::all, 0) - center_x) + sin_theta * (xy(Eigen::all, 1) - center_y));
  a = a * a;
  ArrayX<scalar_T> b = (sin_theta * (xy(Eigen::all, 0) - center_x) - cos_theta * (xy(Eigen::all, 1) - center_y));
  b = b * b;
  ArrayX<scalar_T> ellipse_equation = (a / (radius_major * radius_major)) + (b / (radius_minor * radius_minor));

  return ellipse_equation <= 1;
}

}  // namespace PointTree

#endif  // POINTS_IN_ELLIPSE_H