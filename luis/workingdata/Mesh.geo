// Gmsh project created on Fri Sep 16 12:56:59 2022
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 1, 0, 0.1};
Point(2) = {0, 0, 0, 0.1};
Point(3) = {1, 0, 0, 0.1};
Circle(1) = {1, 2, 3};
Line(2) = {3, 2};
Line(3) = {2, 1};
Curve Loop(1) = {1, 2, 3};
Plane Surface(1) = {1};
Physical Surface(3) = {1};
Physical Curve(1) = {2};
Physical Curve(2) = {3};
MeshSize {2} = 0.001;