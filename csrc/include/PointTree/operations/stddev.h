#include "../type_aliases.h"

#ifndef STDDEV_H
#define STDDEV_H

namespace PointTree {

template <typename scalar_T>
double stddev(RefArrayX2<scalar_T> x) {
  double mean = x.mean();
  double variance = (x - mean).square().mean();
  double stddev = std::sqrt(variance);

  return stddev;
}

}  // namespace PointTree

#endif  // STDDEV_H