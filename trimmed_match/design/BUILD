# The library used to automate trimmed match design for GeoX.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "common_classes",
    srcs = [
        "common_classes.py",
    ],
    srcs_version = "PY3",
)

py_library(
    name = "geo_assignment",
    srcs = [
        "geo_assignment.py",
    ],
    srcs_version = "PY3",
    deps = ["//trimmed_match:estimator"],
)

py_library(
    name = "trimmed_match_design",
    srcs = [
        "trimmed_match_design.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":common_classes",
        ":geo_assignment",
        ":matched_pairs_rmse",
        ":plot_utilities",
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
    ],
)

py_library(
    name = "util",
    srcs = [
        "util.py",
    ],
    srcs_version = "PY3",
    deps = [":common_classes"],
)

py_test(
    name = "common_classes_test",
    size = "small",
    srcs = [
        "tests/common_classes_test.py",
    ],
    python_version = "PY3",
    deps = [":common_classes"],
)

py_test(
    name = "geo_assignment_test",
    size = "small",
    srcs = [
        "tests/geo_assignment_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":geo_assignment",
        "//trimmed_match:estimator",
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
    ],
)

py_test(
    name = "trimmed_match_design_test",
    size = "medium",
    srcs = [
        "tests/trimmed_match_design_test.py",
    ],
    data = ["//trimmed_match/example_datasets:test_data"],
    python_version = "PY3",
    deps = [
        ":common_classes",
        ":matched_pairs_rmse",
        ":trimmed_match_design",
    ],
)

py_test(
    name = "util_test",
    size = "small",
    srcs = [
        "tests/util_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":common_classes",
        ":util",
    ],
)
