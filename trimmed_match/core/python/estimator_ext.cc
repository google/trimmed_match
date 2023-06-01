/*
* Copyright 2020 Google LLC.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ============================================================================
*/
#include "trimmed_match/core/estimator.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace trimmedmatch {

namespace py = pybind11;

PYBIND11_MODULE(estimator_ext, m) {
  py::class_<TrimAndError>(m, "TrimAndError")
      .def(py::init<>())
      .def_readwrite("trim_rate", &TrimAndError::trim_rate)
      .def_readwrite("iroas", &TrimAndError::iroas)
      .def_readwrite("std_error", &TrimAndError::std_error);

  py::class_<Result>(m, "Result")
      .def(py::init<>())
      .def_readwrite("estimate", &Result::estimate)
      .def_readwrite("std_error", &Result::std_error)
      .def_readwrite("trim_rate", &Result::trim_rate)
      .def_readwrite("normal_quantile", &Result::normal_quantile)
      .def_readwrite("conf_interval_low", &Result::conf_interval_low)
      .def_readwrite("conf_interval_up", &Result::conf_interval_up)
      .def_readwrite("candidate_results", &Result::candidate_results);

  py::class_<TrimmedMatch>(m, "TrimmedMatch")
      .def(py::init<const std::vector<double>&, const std::vector<double>&,
                    const double>(),
           py::arg("delta_response"), py::arg("delta_cost"),
           py::arg("max_trim_rate") = 0.25)
      .def("Report", &TrimmedMatch::Report,
           py::arg("normal_quantile") = 1.281551566,
           py::arg("trim_rate") = -1.0);
}

}  // namespace trimmedmatch
