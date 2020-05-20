load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository",
     "new_git_repository")

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

# Bazel rules for Boost C++
git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "135d46b4c9423ee7d494c78a21ff621bc73c12f3",
    remote = "https://github.com/nelhage/rules_boost",
    shallow_since = "1588368648 -0700",
)
load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

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

#
# ======== Python libraries ========
#

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "rules_python",
    remote = "https://github.com/oapio/rules_python.git",
    branch = "master"
)
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

load("@rules_python//python:pip.bzl", "pip_repositories")
pip_repositories()

http_archive(
    name = "six_archive",
    build_file = "@//:external/BUILD.six",
    sha256 = "d16a0141ec1a18405cd4ce8b4613101da75da0e9a7aec5bdd4fa804d0e0eba73",
    strip_prefix = "six-1.12.0",
    urls = [
        "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.12.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.12.0.tar.gz",  # 2018-12-10
    ],
)
 
load("@rules_python//python:pip.bzl", "pip3_import")
pip3_import(
   name = "my_deps",
   requirements = "//:requirements.txt",
)
load("@my_deps//:requirements.bzl", "pip_install")
pip_install()

# Abseil python library
http_archive(
    name = "com_google_absl_py",
    sha256 = "3d0f39e0920379ff1393de04b573bca3484d82a5f8b939e9e83b20b6106c9bbe",
    strip_prefix = "abseil-py-pypi-v0.7.1",
    urls = [
        "http://mirror.bazel.build/github.com/abseil/abseil-py/archive/pypi-v0.7.1.tar.gz",
        "https://github.com/abseil/abseil-py/archive/pypi-v0.7.1.tar.gz",  # 2019-03-12
    ],
)
