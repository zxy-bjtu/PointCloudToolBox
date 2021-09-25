function [] = write_binvox( binvox_filename, vol, t, s )
% WRITE_BINVOX Convert binary voxel occupancy array to .binvox file
% Input
%   binvox_filename - name of .binvox file
%   vol - 3D binary voxel occupancy array
%   t - translation of the center of the volume to world coordinate system 
%   s - scaling of the volume to world coordinate system 
%   (to better understand the meaning of t and s - see
%   http://www.patrickmin.com/binvox/binvox.html)
%
% Copyright (c) 2017 Anastasia Dubrovina. All rights reserved.

if (nargin < 3 || isempty(t))
    t = [0,0,0];
end
if (nargin < 4 || isempty(s))
    s = 1;
end

vol = ipermute(vol,[3 1 2]);  
sz = size(vol);
vol = vol(:);

% create .binvox file
fid = fopen(binvox_filename,'w');

% write header
fprintf(fid,'#binvox 1\n');
fprintf(fid,'dim %d %d %d\n',sz(1),sz(2),sz(3));
fprintf(fid,'translate %.6f %.6f %.6f\n',t(1),t(2),t(3));
fprintf(fid,'scale %.6f\n',s);
fprintf(fid,'data\n');

% create data array (see http://www.patrickmin.com/binvox/binvox.html)
data = [];
curr_val = vol(1); curr_start_pos = 1;
for k = 1:length(vol)
    if (vol(k) ~= curr_val) || (k-curr_start_pos == 255)
        data(end+1) = curr_val; %#ok<AGROW>
        data(end+1) = k-curr_start_pos; %#ok<AGROW>
        curr_val = vol(k);
        curr_start_pos = k;
    end
end
data(end+1) = curr_val;
data(end+1) = k-curr_start_pos+1;

% write data
fwrite(fid,data,'uint8');
fclose(fid);

end
