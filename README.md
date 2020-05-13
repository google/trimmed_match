# Trimmed Match: a robust statistical technique for measuring ad effectiveness through the design and analysis of randomized geo experiments

Copyright (C) 2020 Google LLC. License: Apache 2.0

## Disclaimer

This is not an officially supported Google product. For research purposes only.

## Description

How to properly measure the effectiveness of online advertising (e.g. search, display, video, etc) is a fundamental problem not only for advertisers but for Google. Randomized geo experiments (Vaver & Koehler, 2010) have been recognized as the gold standard for the measurements, but how to design and analyze them properly is a non-trivial statistical problem. Unlike the usual A/B tests, in GeoX, the number of geos is usually small; Moreover, there is often severe heterogeneity across geos, which makes traditional regression adjustment less reliable. Furthermore, due to temporal dynamics, geos between the treatment group and the control group may become less comparable during the test period even if they were comparable during the design phase, which is often obvious by looking at the time period after the design was done but before the experiment started. In order to address these technical issues, Trimmed Match (Chen & Au, 2019) has recently been developed to improve existing methodologies in analyzing randomized paired geo experiments. We also apply Trimmed Match and cross validation to improve the traditional design of matched pairs.

This directory contains

  * C++ core library and Python wrapper for Trimmed Match, and
  * Python package for geox design using Trimmed Match and Cross Validation.
  * (TBD) Colab demos for trimmed match design and post analysis, separately.

## Installation
Our current version has been tested with python3 under Linux.

### Prerequisites
  * The build tool `bazel`: see the instruction at https://github.com/bazelbuild/bazel.

  * Installation of `pip`:

  ```
  sudo apt-get install python3-pip
  ```

* Python libraries:

  ```
  python3 -m pip install --user absl-py dataclasses matplotlib numpy pandas pybind11 seaborn setuptools six
  ```

### Trimmed Match can be installed using `pip`

First clone from github:

```
git clone https://github.com/google/trimmed_match
```

To make sure the package and all dependencies are installed properly, run

```
bazel test ... --action_env=PYTHON_BIN_PATH=/usr/bin/python3
```

Then `pip` install:

```
python3 setup.py bdist_wheel

python3 -m pip install dist/trimmed_match*
```

## Usage

Without programming, the best way to learn how to use the package is to follow one of the
notebooks (<span style="color:blue">to be added</span>), and the recommended way of opening them is Google `Colab`.

 * Design a matched pairs geo experiment.
    - [GeoX Design with Trimmed Match and Cross Validation](./notebooks/design_colab_for_trimmed_match.ipynb)
 * Post analysis of a geo experiment from a matched pairs design
    - [GeoX Post Analysis with Trimmed Match](./notebooks/post_analysis_colab_for_trimmed_match.ipynb)


With Python programming, here is a toy example.

```python
import numpy as np
import pandas as pd
import trimmed_match
from trimmed_match.estimator import Report, TrimmedMatch
from trimmed_match.design.common_classes import GeoXType, TimeWindow
from trimmed_match.design.trimmed_match_design import TrimmedMatchGeoXDesign


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


#####################################################################
# Designs a trimmed matched pairs geo experiment
#####################################################################

pretest_data = pd.DataFrame(data={
   'date': ['2019-01-01','2019-03-01'] * 30,
   'geo': np.repeat(range(30), 2),
   'sales': range(60),
   'cost': range(60),
   'transactions': range(60),
})

response = 'sales'
spend_proxy = 'cost'
matching_metrics = {'sales':1.0, 'transactions':1.0, 'cost':.01}
time_window_for_design = TimeWindow('2019-01-01', '2019-03-01')
time_window_for_eval = TimeWindow('2019-02-01', '2019-03-01')

tmd = TrimmedMatchGeoXDesign(
   GeoXType.HEAVY_UP, pretest_data, response, matching_metrics,
   time_window_for_design, time_window_for_eval)

budget_list = [100.0]
hypothesized_iroas_list = [1.0]
candidate_designs = tmd.report_candidate_designs(
   budget_list, hypothesized_iroas_list, spend_proxy,
   num_pairs_filtered_list=[0])

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
