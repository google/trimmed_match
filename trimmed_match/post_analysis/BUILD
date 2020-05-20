# The library used to automate trimmed match post analysis for GeoX.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "trimmed_match_post_analysis",
    srcs = [
        "trimmed_match_post_analysis.py",
    ],
    srcs_version = "PY3",
    deps = [
        requirement("pandas"),
        "//trimmed_match:estimator",
        "//trimmed_match/design:util",
    ],
)

py_test(
    name = "trimmed_match_post_analysis_test",
    size = "small",
    srcs = [
        "tests/trimmed_match_post_analysis_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":trimmed_match_post_analysis",
        requirement("numpy"),
        requirement("pandas"),
        "//trimmed_match:estimator",
        "//trimmed_match/design:plot_utilities",
    ],
)
