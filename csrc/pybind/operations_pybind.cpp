#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/PointTree/operations/fit_ellipse.h"
#include "../include/PointTree/operations/points_in_ellipse.h"

PYBIND11_MODULE(_operations_cpp, m) {
  m.doc() = R"pbdoc(
    C++ extension module implementing operations for point cloud processing.
  )pbdoc";

  m.def("fit_ellipse", &PointTree::fit_ellipse<float>, pybind11::return_value_policy::reference_internal, "");

  m.def("fit_ellipse", &PointTree::fit_ellipse<double>, pybind11::return_value_policy::reference_internal, R"pbdoc(
    Fits an ellipse to a set of 2D points using a least-squares method. For more details, see the documentation of the
    Python wrapper method :code:`pointtree.operations.fit_ellipse`.
  )pbdoc");

  m.def(
      "points_in_ellipse", &PointTree::points_in_ellipse<float>, pybind11::return_value_policy::reference_internal, "");

  m.def(
      "points_in_ellipse", &PointTree::points_in_ellipse<double>, pybind11::return_value_policy::reference_internal,
      R"pbdoc(
    Tests whether 2D points are within the boundaries of an ellipse. For more details, see the documentation of the
    Python wrapper method :code:`pointtree.operations.points_in_ellipse`.
  )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = (VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
