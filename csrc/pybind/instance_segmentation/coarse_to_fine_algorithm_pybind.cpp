#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../include/PointTree/instance_segmentation/coarse_to_fine_algorithm.h"

PYBIND11_MODULE(_coarse_to_fine_algorithm_cpp, m) {
  m.doc() = R"pbdoc(
    C++ extension module implementing selected steps of the CoarseToFineAlgorithm.
  )pbdoc";

  m.def("grow_trees", &PointTree::grow_trees<float>, pybind11::return_value_policy::reference_internal, "");

  m.def(
      "grow_trees", &PointTree::grow_trees<double>, pybind11::return_value_policy::reference_internal,
      R"pbdoc(
    C++ implementation of the region-growing method for tree crown segmentation. For more details, see the documentation
    of the Python wrapper method :code:`CoarseToFineAlgorithm.grow_trees()`.
  )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = (VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
