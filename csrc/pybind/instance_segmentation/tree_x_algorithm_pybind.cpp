#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../instance_segmentation/tree_x_algorithm.h"

PYBIND11_MODULE(_tree_x_algorithm_cpp, m) {
  m.doc() = R"pbdoc(
    C++ extension module implementing selected steps of the TreeXAlgorithm.
  )pbdoc";

  m.def("collect_inputs_trunk_layers_preliminary_fitting", &collect_inputs_trunk_layers_preliminary_fitting,
        pybind11::return_value_policy::reference_internal,
        R"pbdoc(
    In the trunk identification, horizontal layers are extracted from the potential trunk clusters and circles or
    ellipses are fitted to these layers. Initially, the circle / ellipse fitting is done using the downsampled points
    assigned to the individual clusters. For this preliminary circle / ellipse fitting, this method compiles the points
    (and the indices of these points) within the respective layer for each trunk cluster and layer.
  )pbdoc");

  m.def("collect_inputs_trunk_layers_exact_fitting", &collect_inputs_trunk_layers_exact_fitting,
        pybind11::return_value_policy::reference_internal, R"pbdoc(
    In the trunk identification, horizontal layers are extracted from the potential trunk clusters and circles /
    ellipses are fitted to these layers. After fitting preliminary circles or ellipses to each layer, the points within
    a buffer region around the outline of each circle / ellipse are extracted and the circle / ellipse fitting is
    repeated using theses points. For this more exact circle / ellipse fitting, this method compiles the points (and the
    indices of these points) within the respective buffer region for each trunk cluster and layer.
  )pbdoc");

  m.def("collect_region_growing_seeds", &collect_region_growing_seeds,
        pybind11::return_value_policy::reference_internal, R"pbdoc(
    Method that collects the seed points for tree crown segmentation using region growing.
  )pbdoc");

  m.def("segment_tree_crowns", &segment_tree_crowns, pybind11::return_value_policy::reference_internal, R"pbdoc(
    C++ implementation of the region-growing method for tree crown segmentation. For more details, see the documentation
    of the Python wrapper method :code:`TreeXAlgorithm.segment_tree_crowns()`.
  )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = (VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
