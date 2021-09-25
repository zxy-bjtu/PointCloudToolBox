function [ V, TRI ] = visualize_voxels( binary_vol )
%visualize_voxels Summary of this function goes here
%   Detailed explanation goes here

inds = find(binary_vol);
[Y,X,Z] = ind2sub(size(binary_vol),inds);
cube_verts = [0 0 0;
              1 0 0;
              0 1 0;
              1 1 0;
              0 0 1;
              1 0 1;
              0 1 1;
              1 1 1];
cube_tri = [1 2 3;
            2 3 4;
            1 2 5;
            2 5 6;
            1 3 5;
            3 5 7;
            3 4 8;
            3 8 7;
            2 4 8;
            2 8 6;
            5 8 6;
            5 7 8];

V = [];
TRI = [];

% For each voxel center (X(k),Y(k),Z(k)), create a 1x1x1 meshed cube around it 
for k = 1:length(X)
    V_k = bsxfun(@plus,cube_verts,[X(k),Y(k),Z(k)]-0.5);
    V = [V; V_k];
    TRI = [TRI; cube_tri+8*(k-1)];
end

% figure;
dims = size(binary_vol);
trisurf(TRI, V(:,1), V(:,2), V(:,3)); 
% axis image; 
% axis([1 dims(2) 1 dims(1) 1 dims(3)]);
axis off;

end
