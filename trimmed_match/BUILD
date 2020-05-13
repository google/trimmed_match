licenses(["notice"])

exports_files(["LICENSE"])

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
    deps = [
        ":estimator",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)
