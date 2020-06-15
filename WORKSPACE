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
  commit = "96a2f23dca4cc7180821ca5f32e526314395d26a",
  shallow_since = "1553223106 +0900",
  remote = "https://github.com/google/glog",
)

# NOTE: We directly depend on glog, which transitively depends on gflags.
# Bazel doesn't download transitive dependencies so we have to include it
# directly.
# https://docs.bazel.build/versions/master/external.html#transitive-dependencies
git_repository(
  name = "com_github_gflags_gflags",
  commit = "e171aa2d15ed9eb17054558e0b3a6a413bb01067",
  remote = "https://github.com/gflags/gflags",
)

# Google C++ Testing Framework
git_repository(
  name = "com_google_googletest",
  tag = "release-1.10.0",
  remote = "https://github.com/google/googletest",
)

# pybind11
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    strip_prefix = "pybind11-2.4.3",
    urls = ["https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz"],
)

# Bazel rules for pybind11
http_archive(
    name = "pybind11_bazel",
    strip_prefix = "pybind11_bazel-d5587e65fb8cbfc0015391a7616dc9c66f64a494",
    url = "https://github.com/pybind/pybind11_bazel/archive/d5587e65fb8cbfc0015391a7616dc9c66f64a494.zip",
    sha256 = "bf8e1f3ebde5ee37ad30c451377b03fbbe42b9d8f24c244aa8af2ccbaeca7e6c",
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")
