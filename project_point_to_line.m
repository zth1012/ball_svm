function [proj_x, proj_y] = project_point_to_line(line_params, q_x, q_y)

% Line equation: ax + by + c = 0
a = line_params(1);
b = line_params(2);
c = line_params(3);

% Get two points on the line
p0_x = 0;
p0_y = -c/b;

p1_x = 1;
p1_y = -(a + c)/b;

A = [p1_x - p0_x, p1_y - p0_y; p0_y - p1_y, p1_x - p0_x];
B = -[-q_x*(p1_x-p0_x)-q_y*(p1_y-p0_y);-p0_y*(p1_x-p0_x)+p0_x*(p1_y-p0_y)];

X = A\B;
proj_x = X(1);
proj_y = X(2);
