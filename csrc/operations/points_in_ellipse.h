#include <Eigen/Dense>
#include <vector>

Eigen::Array<bool, Eigen::Dynamic, 1> points_in_ellipse(Eigen::ArrayX2d xy,
                                                        Eigen::Array<double, Eigen::Dynamic, 1> ellipse);
