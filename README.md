# Routing Problems for Multiple Cooperative UAVs using Transformers

## Paper
Solving a variant of the Orieentering Problem (OP) called the Orienteering Problem with Multiple Prizes and Types of
Node (OP-MP-TN) with a cooperative multi-agent system based on Transformer Networks. For more details, please see our
[paper](). If this code is useful for your work, please cite our paper:

```

``` 

## Dependencies

* Python >= 3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/) >= 1.7
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib
* [k-means-constrained](https://joshlk.github.io/k-means-constrained/)

## Usage

First, it is necessary to create training, testing, and validation datasets:
```bash
python create_dataset.py --graph_sizes 20 --train_sizes 1280000 --test_sizes 10000 --val_sizes 10000
```

Datasets with multiple sizes can be created with:
```bash
python create_dataset.py --graph_sizes 20 50 --train_sizes 640000 640000 --test_sizes 5000 5000 --val_sizes 5000 5000
```

To train a Transformer model(attention) use:
```bash
python run.py --model attention --graph_size 20 --train_dataset --train_dataset datasets/op/coop/4agents/1depots/20_seed1234/train --val_dataset datasets/op/coop/4agents/1depots/20_seed1234/val
```

Pointer Network (pointer) and Graph Pointer Network (gpn) can also be trained with the `--model` option. To resume
training, load your last saved model with the `--resume` option.

Evaluate your trained models with:
```bash
python eval.py --model outputs/op_coop20/attention_run --test_dataset datasets/op/coop/4agents/1depots/20_seed1234/test/20/data.pkl
```
If the epoch is not specified, by default the last one in the folder will be used.

Baselines like [OR-Tools](https://developers.google.com/optimization), [Gurobi](https://www.gurobi.com),
[Tsiligirides](https://www.tandfonline.com/doi/abs/10.1057/jors.1984.162),
[Compass](https://github.com/bcamath-ds/compass) or a [Genetic Algorithm](https://github.com/mc-ride/orienteering) can
be executed as follows:
```bash
python -m problems.op.op_baseline --method ortools --datasets datasets/op/coop/4agents/1depots/20_seed1234/test/20/data.pkl
```
To run Compass, you need to install it by running the `install_compass.sh` script from within the `problems/op`
directory. To use Gurobi, obtain a ([free academic](http://www.gurobi.com/registration/academic-license-reg)) license
and follow the
[installation instructions](https://www.gurobi.com/documentation/8.1/quickstart_windows/installing_the_anaconda_py.html)
. OR-Tools has to be installed too (`pip install ortools`).

### Other options and help
```bash
python run.py -h
python eval.py -h
```

## Acknowledgements
This repository is an adaptation of
[wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) for the case of multiple
cooperative UAVs.
