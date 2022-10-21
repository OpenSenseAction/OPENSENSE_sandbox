#!/bin/bash

echo 'Running with the following python binary'
which python

cd ..

echo ''
echo 'get content of submodules...'
git submodule update --init

echo ''
echo 'add python submodules to conda env...'
conda develop pycomlink
conda develop PyNNcml
# not sure if we need this one since it currently is not 
# used in the example notebook of Abbas which defines all
# required functions inside the notebook
conda develop pws-pyqc

echo ''
echo 'Install RAINLINK...'
cd RAINLINK
R CMD INSTALL RAINLINK*.tar.gz

echo ''
echo 'create dir in RAINLINK for figures...'
mkdir Figures

echo ''
echo '...done'