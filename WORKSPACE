load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

#
# ======== C++ LIBRARIES ========
#

# Abseil
git_repository(
  name = "com_google_absl",
  tag = "20190808",
  remote = "https://github.com/abseil/abseil-cpp",
)

# Google Logging Library
git_repository(
  name = "glog",
  commit = "9630e0e848da22e27b346c38d9b05f0a16cbf7b3",
  shallow_since = "1570114335 -0400",
  remote = "https://github.com/google/glog",
)

# NOTE: We directly depend on glog, which transitively depends on gflags.
# Bazel doesn't download transitive dependencies so we have to include it
# directly.
# https://docs.bazel.build/versions/master/external.html#transitive-dependencies
git_repository(
  name = "com_github_gflags_gflags",
  commit = "a738fdf9338412f83ab3f26f31ac11ed3f3ec4bd",
  remote = "https://github.com/gflags/gflags",
)

# Google C++ Testing Framework
git_repository(
  name = "com_google_googletest",
  commit = "5c08f92c881b666998a4f7852c3cf9e393bf33a7",
  shallow_since = "1615772488 -0700",
  remote = "https://github.com/google/googletest",
)

# pybind11
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    strip_prefix = "pybind11-2.10.3",
    urls = ["https://github.com/pybind/pybind11/archive/v2.10.3.tar.gz"],
)

# Bazel rules for pybind11
http_archive(
    name = "pybind11_bazel",
    strip_prefix = "pybind11_bazel-9a24c33cbdc510fa60ab7f5ffb7d80ab89272799",
    url = "https://github.com/pybind/pybind11_bazel/archive/9a24c33cbdc510fa60ab7f5ffb7d80ab89272799.zip",
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

# Bazel Skylib library required for Absl C++ library
http_archive(
    name = "bazel_skylib",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()