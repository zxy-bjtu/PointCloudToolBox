function [ vol, t, s ] = read_binvox( binvox_filename, do_visualize )
% READ_BINVOX Read binvox file contents and convert to 3D binary voxel occupancy array
% Input
%   binvox_filename - name of .binvox file
%   do_visualize - visualization flag. When 'true' - plot the array using
%   'visualize_voxels' function
% Output
%   vol - 3D binary voxel occupancy array
%   t - translation of the center of the volume to world coordinate system 
%   s - scaling of the volume to world coordinate system 
%   (to better understand the meaning of t and s - see
%   http://www.patrickmin.com/binvox/binvox.html)
%
% Copyright (c) 2017 Anastasia Dubrovina. All rights reserved.

% visualizattion flag is false by default
if (nargin < 2)
    do_visualize = false;
end

% read data
fid = fopen(binvox_filename);
fgetl(fid); % '#binvox 1'
l = fgetl(fid); % e.g. 'dim 64 64 64'
dims = sscanf(l, 'dim %d %d %d');
l = fgetl(fid); % e.g. 'translate -1.74325 -0.929973 -1.21382'
t = sscanf(l, 'translate %f %f %f');
l = fgetl(fid); % e.g. 'scale 3.48649'
s = sscanf(l, 'scale %f');
fgetl(fid); % 'data'
data = fread(fid);
fclose(fid);

% create volume
vol = zeros(dims(:)');
k = 1; curr_ind = 0;
while k < length(data)
    val = data(k);
    len = data(k+1);
    k = k+2;
    vol(curr_ind+(1:len)) = val;
    curr_ind = curr_ind+len;
end
vol = permute(vol,[3 1 2]); %[y x z]

% visualize
if (do_visualize)
    figure; visualize_voxels(vol); axis([0 dims(1)+1 0 dims(2)+1 0 dims(3)+1]); cameratoolbar;
end

end

