# TDMSci corpus snapshot

This directory contains the authors' TDMSci v2 CoNLL train and test files from
the archived `IBM/science-result-extractor` repository. They were retrieved
from [Software Heritage's archived repository snapshot](https://archive.softwareheritage.org/api/1/snapshot/7e6e62500efac3476e3b72a132df9da899b33d75/),
revision `1c59ff337d09a6c3af313c65cf1c43a5b7f89c7a` (2021-05-20), because
the [original repository](https://github.com/IBM/science-result-extractor) was
unavailable when this snapshot was prepared. The files are unchanged.

| File | SHA-256 | Parsed sentences |
| --- | --- | ---: |
| `train_1500_v2.conll` | `8107bc8a0214f3c57d234e2ab69958c1d4d8b11e43e2007d97ea42efaf7a231a` | 1,523 |
| `test_500_v2.conll` | `5d776e10f080c2872bba416716bc2109a86946974942a255b8dda3dd78078559` | 487 |

The parsed sentence counts differ from the 1,500/500 names and paper counts, but the
entity counts match those reported in the [TDMSci paper](https://aclanthology.org/2021.eacl-main.59/):
train 1,219 task, 420 dataset and 536 metric mentions; test 396 task,
192 dataset and 174 metric mentions. We use the published BIO tags directly.

Using the exact ordered token sequence as the identity key, the training file
has 1,517 unique sequences among 1,523 rows (six repeated rows; three groups
have different annotation sets). The test file has 484 unique sequences among
487 rows (three repeated rows, with consistent annotations). Ten distinct test
sequences also occur in training; five have different annotation sets across
the splits. These are properties of the archived release; the files above have
not been edited or deduplicated.

Original copyright: IBM Corporation 2019. The original Apache 2.0 license is
included as `LICENSE`. Cite Hou et al., *TDMSci: A Specialized Corpus for
Scientific Literature Entity Tagging of Tasks Datasets and Metrics*, EACL 2021.
