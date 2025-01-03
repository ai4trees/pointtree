#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace {
using namespace Eigen;

using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;
}  // namespace

ArrayXb points_in_ellipse(ArrayX2d xy, ArrayXd ellipse) {
  double center_x = ellipse(0);
  double center_y = ellipse(1);
  double radius_major = ellipse(2);
  double radius_minor = ellipse(3);
  double theta = ellipse(4);

  double cos_theta = cos(theta);
  double sin_theta = sin(theta);

  ArrayXd a = (cos_theta * (xy(Eigen::all, 0) - center_x) + sin_theta * (xy(Eigen::all, 1) - center_y));
  a = a * a;
  ArrayXd b = (sin_theta * (xy(Eigen::all, 0) - center_x) - cos_theta * (xy(Eigen::all, 1) - center_y));
  b = b * b;
  ArrayXd ellipse_equation = (a / (radius_major * radius_major)) + (b / (radius_minor * radius_minor));

  return ellipse_equation <= 1;
}
