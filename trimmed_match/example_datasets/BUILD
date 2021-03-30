package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

py_binary(
    name = "simulation_script",
    srcs = ["simulation_script.py"],
    python_version = "PY3",
)

filegroup(
    name = "test_data",
    srcs = glob(["*.csv"]),
)
