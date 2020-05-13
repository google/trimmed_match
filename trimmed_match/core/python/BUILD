# PYBIND11 wrapper for trimmedmatch package.

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

licenses(["notice"])

exports_files(["LICENSE"])

pybind_extension(
    name = "estimator_ext",
    srcs = ["estimator_ext.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = [
        "//trimmed_match:__subpackages__",
    ],
    deps = ["//trimmed_match/core:estimator"],
)
