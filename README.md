# cuDCNv2

A full cuda implementation of dcnv2 forward, without dependent of cutorch. For easily integrated into c++ project.

Derived from [CharlesShang/DCNv2](https://github.com/CharlesShang/DCNv2).

## prepare weights

get `dcnv2.wts` by following steps.

```
git clone https://github.com/wang-xinyu/DCNv2.git
git checkout pytorch-0.4
./make.sh
python test_dcn_func.py
```

## run

```
mkdir build
cd build
cmake ..
make
sudo ./app
```
