%% 将binvox转成vox并进行可视化
clc;
clear all;
binvox_filename = './chair.binvox';
do_visualize = true;
read_binvox(binvox_filename, do_visualize);
