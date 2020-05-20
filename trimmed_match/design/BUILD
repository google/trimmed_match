# The library used to automate trimmed match design for GeoX.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

load("@my_deps//:requirements.bzl", "requirement")

py_library(
    name = "common_classes",
    srcs = [
        "common_classes.py",
    ],
    srcs_version = "PY3",
    deps = [
        requirement("pandas"),
    ],
)

py_library(
    name = "trimmed_match_design",
    srcs = [
        "trimmed_match_design.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":common_classes",
        ":matched_pairs_rmse",
        ":plot_utilities",
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pandas"),
        requirement("seaborn"),
    ],
)

py_library(
    name = "matched_pairs_rmse",
    srcs = [
        "matched_pairs_rmse.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":common_classes",
        requirement("numpy"),
        requirement("pandas"),
        requirement("six"),
        "//trimmed_match:estimator",
    ],
)

py_library(
    name = "plot_utilities",
    srcs = [
        "plot_utilities.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":common_classes",
        ":util",
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pandas"),
        requirement("seaborn"),
    ],
)

py_library(
    name = "util",
    srcs = [
        "util.py",
    ],
    srcs_version = "PY3",
    deps = [
        requirement("numpy"),
        requirement("pandas"),
    ],
)

py_test(
    name = "common_classes_test",
    size = "small",
    srcs = [
        "tests/common_classes_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":common_classes",
        requirement("pandas"),
    ],
)

py_test(
    name = "matched_pairs_rmse_test",
    size = "small",
    srcs = [
        "tests/matched_pairs_rmse_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":common_classes",
        ":matched_pairs_rmse",
        requirement("pandas"),
    ],
)

py_test(
    name = "trimmed_match_design_test",
    size = "small",
    srcs = [
        "tests/trimmed_match_design_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":common_classes",
        ":matched_pairs_rmse",
        ":trimmed_match_design",
        requirement("matplotlib"),
        requirement("pandas"),
    ],
)

py_test(
    name = "util_test",
    size = "small",
    srcs = [
        "tests/util_test.py",
    ],
    python_version = "PY3",
    deps = [":util"],
)
