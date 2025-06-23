#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/PointTree/evaluation/instance_segmentation_metrics.h"

PYBIND11_MODULE(_evaluation_cpp, m) {
  m.doc() = R"pbdoc(
    C++ extension module implementing evaluation functions.
  )pbdoc";

  m.def(
      "compute_instance_segmentation_metrics", &PointTree::compute_instance_segmentation_metrics<float>,
      pybind11::return_value_policy::reference_internal, "");

  m.def(
      "compute_instance_segmentation_metrics", &PointTree::compute_instance_segmentation_metrics<double>,
      pybind11::return_value_policy::reference_internal,
      R"pbdoc(Computes metrics to measure the quality of the point-wise segmentation.)pbdoc");

  m.def(
      "compute_instance_segmentation_metrics_per_partition",
      &PointTree::compute_instance_segmentation_metrics_per_partition<float>,
      pybind11::return_value_policy::reference_internal, "");

  m.def(
      "compute_instance_segmentation_metrics_per_partition",
      &PointTree::compute_instance_segmentation_metrics_per_partition<double>,
      pybind11::return_value_policy::reference_internal,
      R"pbdoc(Calculates instance segmentation metrics for different spatial partitions of a tree instance.)pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = (VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
