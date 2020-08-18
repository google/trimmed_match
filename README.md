# Trimmed Match: a robust statistical technique for measuring ad effectiveness through the design and analysis of randomized geo experiments

Copyright (C) 2020 Google LLC. License: Apache 2.0

## Disclaimer

This is not an officially supported Google product. For research purposes only.

## Description

How to properly measure the effectiveness of online advertising (e.g. search, display, video, etc) is a fundamental problem not only for advertisers but for Google. Randomized geo experiments (Vaver & Koehler, 2010) have been recognized as the gold standard for the measurements, but how to design and analyze them properly is a non-trivial statistical problem. Unlike the usual A/B tests, in GeoX, the number of geos is usually small; Moreover, there is often severe heterogeneity across geos, which makes traditional regression adjustment less reliable. Furthermore, due to temporal dynamics, geos between the treatment group and the control group may become less comparable during the test period even if they were comparable during the design phase, which is often obvious by looking at the time period after the design was done but before the experiment started. Trimmed Match (Chen & Au, 2019) has recently been developed in order to address these technical issues in analyzing randomized paired geo experiments. We also apply Trimmed Match and cross validation to improve the traditional design of matched pairs.

This version contains

  * C++ core library and Python wrapper for Trimmed Match, and
  * Python package for geo experimental design (preliminary version) using Trimmed Match and Cross Validation.
  * Corresponding colab demos for post analysis and experimental design, separately.

## Installation

Our current version has been tested with Python 3.7 in Linux. The code may be incompatible with Python 3.6 or lower versions.

### Prerequisites

  * A Python 3 development environment with `setuptools` and `git`:

  ```
  sudo apt-get install python3-dev python3-setuptools git
  ```

  * The build tool `bazel`: Installation instructions can be found at
  https://github.com/bazelbuild/bazel.

### Trimmed Match can be installed using `setuptools` and `pip`

First clone from github:

```shell
git clone https://github.com/google/trimmed_match
```

Then build and install the extension using the supplied `setup.py` file and `setuptools` and `pip`.

```
python3 -m pip install ./trimmed_match
```

This will automatically build the extension using `bazel`.

You can run the unit tests using:

```
cd trimmed_match
PYTHON_BIN_PATH=`which python3` bazel test //...:all

```
Note that unit tests require package dependencies to be installed.
This automatically happens when the package is installed as above using `pip`.
Otherwise, the dependencies can be installed manually using the following
command:

```
python3 -m pip install absl-py matplotlib numpy pandas seaborn scipy
```

## Usage

Without programming, the best way to learn how to use the package is to follow one of the
notebooks, and the recommended way of opening them is Google `Colab`.

 * Post analysis of a geo experiment from a matched pairs design
    - [GeoX Post Analysis with Trimmed Match](https://colab.sandbox.google.com/github/google/trimmed_match/blob/master/trimmed_match/notebook/post_analysis_colab_for_trimmed_match.ipynb)
 * Design a matched pairs geo experiment (preliminary version)
    - [GeoX Design with Trimmed Match and Cross Validation](https://colab.sandbox.google.com/github/google/trimmed_match/blob/master/trimmed_match/notebook/design_colab_for_trimmed_match.ipynb)


With Python programming, here is a toy example.

```python
import trimmed_match
from trimmed_match.estimator import TrimmedMatch


#####################################################################
# Reports the estimate of incremental return on ad spend (iROAS)
# using geo experimental data from a matched pairs design (5 geo pairs)
#####################################################################

delta_response = [1, 10, 3, 8, 5]  # response difference between treatment and control in each geo pair
delta_spend = [1, 5, 2, 5, 3]      # spend difference between treatment and control in each geo pair
confidence_level = 0.80            # for the two-sided confidence interval
tm = TrimmedMatch(delta_response, delta_spend)
report = tm.Report(confidence=confidence_level)
print('iroas=%.2f, ci=(%.2f, %.2f)' % (
      report.estimate, report.conf_interval_low, report.conf_interval_up))

# iroas=1.60, ci=(1.52, 1.66)
```

## References

Aiyou Chen and Tim Au (2019). Robust Causal Inference for Incremental Return on Ad Spend with Randomized Geo Experiments.
(https://research.google/pubs/pub48448/)

Jon Vaver and Jim Koehler (2011). Measuring Ad Effectiveness Using Geo Experiments.
(https://research.google/pubs/pub38355/)


## Contact and mailing list

If you want to contribute, please read [CONTRIBUTING](CONTRIBUTING.md)
and send us pull requests. You can also report bugs or file feature requests.

If you'd like to talk to the developers or get notified about major
updates, you may want to subscribe to our
[mailing list](https://groups.google.com/forum/#!forum/trimmed-match-users).


## Developers

* Aiyou Chen
* Marco Longfils
* Christoph Best

