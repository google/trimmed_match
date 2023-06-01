load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

#
# ======== C++ LIBRARIES ========
#

# Abseil
http_archive(
    name = "com_google_absl",
    sha256 = "44634eae586a7158dceedda7d8fd5cec6d1ebae08c83399f75dd9ce76324de40",  # Last updated 2022-05-18
    strip_prefix = "abseil-cpp-3e04aade4e7a53aebbbed1a1268117f1f522bfb0",
    urls = ["https://github.com/abseil/abseil-cpp/archive/3e04aade4e7a53aebbbed1a1268117f1f522bfb0.zip"],
)
# skylib dependency required for Abseil
http_archive(
  name = "bazel_skylib",
  urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz"],
  sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",
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

http_archive(
    name = "pybind11_abseil",
    sha256 = "6481888831cd548858c09371ea892329b36c8d4d961f559876c64e009d0bc630",
    strip_prefix = "pybind11_abseil-3922b3861a2b27d4111e3ac971e6697ea030a36e",
    url = "https://github.com/pybind/pybind11_abseil/archive/3922b3861a2b27d4111e3ac971e6697ea030a36e.tar.gz",
    patches = ["//trimmed_match:status_module.patch"],
)

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