package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "estimator",
    srcs = ["estimator.cc"],
    hdrs = ["estimator.h"],
    deps = [
        ":geox_data_util",
        ":math_util",
        "@com_google_absl//absl/types:optional",
        "@glog",
    ],
)

cc_library(
    name = "geox_data_util",
    srcs = ["geox_data_util.cc"],
    hdrs = ["geox_data_util.h"],
    deps = [
        ":math_util",
        "@glog",
    ],
)

cc_library(
    name = "math_util",
    srcs = ["math_util.cc"],
    hdrs = ["math_util.h"],
    deps = [
        "@glog",
    ],
)

cc_test(
    name = "estimator_test",
    srcs = ["estimator_test.cc"],
    copts = ["-Wno-unused-variable"],
    deps = [
        ":estimator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "geox_data_util_test",
    srcs = ["geox_data_util_test.cc"],
    copts = ["-Wno-unused-variable"],
    deps = [
        ":geox_data_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "math_util_test",
    srcs = ["math_util_test.cc"],
    copts = ["-Wno-unused-variable"],
    deps = [
        ":math_util",
        "@com_google_googletest//:gtest_main",
    ],
)
