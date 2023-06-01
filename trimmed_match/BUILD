licenses(["notice"])

exports_files([
    "LICENSE",
    "example_datasets",
])

py_library(
    name = "estimator",
    srcs = [
        "estimator.py",
    ],
    data = ["//trimmed_match/core/python:estimator_ext.so"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_test(
    name = "estimator_test",
    srcs = ["estimator_test.py"],
    main = "estimator_test.py",
    python_version = "PY3",
    deps = [":estimator"],
)
