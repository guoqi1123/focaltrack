#!bin/bash

## Basic installation
=======================================

# Install and link intel parallel studio
# TODO: Remember to change default installation options to install everything

cd ~/install_dirs/
tar zxvf ~/Downloads/parallel_studio_xe_2016_update2.tgz 
cd parallel_studio_xe_2016_update2/
sudo mkdir /usr/local/intel_parallel_studio_xe_2016_update2
sudo ./install.sh 
cd /usr/local/
sudo ln -sf intel_parallel_studio_xe_2016_update2/ intel
cd /usr/local/share/man
sudo mkdir man1
cd man1
sudo ln -sf /usr/local/intel/man/en_US/man1/* .
sudo mandb
cd
sudo sed -e '/^PATH/s/"$/:\/usr\/local\/intel\/bin"/g' -i /etc/environment
source /etc/environment
echo "/usr/local/intel/lib/intel64" | sudo tee /etc/ld.so.conf.d/icc.conf > /dev/null
echo "/usr/local/intel/ipp/lib/intel64" | sudo tee /etc/ld.so.conf.d/ipp.conf > /dev/null
echo "/usr/local/intel/mkl/lib/intel64" | sudo tee /etc/ld.so.conf.d/mkl.conf > /dev/null
sudo ldconfig

cd /home/qi/anaconda3/lib
mkdir backup
mv libmkl*.so backup/
cd backup/
rename 's/.so/.so.bak/g' libmkl*.so
cd ..
ln -sf /usr/local/intel/mkl/lib/intel64/*.so .

cd /home/qi/anaconda3/envs/tensorflow/lib
mkdir backup
mv libmkl*.so backup/
cd backup/
rename 's/.so/.so.bak/g' libmkl*.so
cd ..
ln -sf /usr/local/intel/mkl/lib/intel64/*.so .


include_dirs = /usr/local/intel/mkl/include/
