load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# PYBIND11 wrapper for trimmedmatch package.
pybind_extension(
    name = "estimator_ext",
    srcs = ["estimator_ext.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "//trimmed_match/core:estimator",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)
