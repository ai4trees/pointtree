#include <Eigen/Dense>
#include <cstdint>
#include <vector>

Eigen::Array<double, Eigen::Dynamic, 5> fit_ellipse(Eigen::ArrayX2d xy,
                                                    Eigen::Array<int64_t, Eigen::Dynamic, 1> batch_lengths,
                                                    int num_workers = 1);
