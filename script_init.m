!python3 main.py


% the array looks like :
% each r1 ,r2,r3,r4,r5 are array of N elements ( N frames) , each element
% display the degrees of motor of the arm.

% t : 0 1  2  3  4  5  6  N
% r1: 0 12 30 50 60 70 80
% r2: 0 80 30 50 60 70 80


%load out.mat r1..N from python script

load("out.mat")
N = length(r2);

t = transpose(0:1:N-1);


r1 = zeros(N,1) + 90; % 90 90 90 90 90 N % r1 rotate top
r5 = zeros(N,1) + 0; % rotate base

r3 = transpose(double(r3));
r2 = transpose(double(r2));
r4 = transpose(double(r4));


% smooth signals with Savitzky-Golay Filters

r2_new   = sgolayfilt(r2, 5, 21);
r3_new   = sgolayfilt(r3, 5, 21);
r4_new   = sgolayfilt(r4, 5, 21);


figure
hold all;

% plot(t,[r2,r2_new]);
% 
% 
% legend("Second arm (r2) degrees ", ...
% "Order 5 - Smooth second arm (r2) degrees");
% 
% title("Second Arm (r2) Degrees Servo By Time");
% 
% xlabel("Frames");
% ylabel("Degrees 0~180");


plot(t,[r3,r3_new]);


legend("Elbow arm (r3) degrees ","Smooth Elbow arm (r3) degrees");

title("Elbow Arm (r3) Degrees Servo By Time");

xlabel("Frames");
ylabel("Degrees 0~180");

% 
% 
% 
% plot(t,[r4,r4_new]);
% 
% 
% legend("Top arm (r4) degrees ","Smooth Top arm (r4) degrees");
% 
% title("Top Arm (r4) Degrees Servo By Time");
% 
% xlabel("Frames");
% ylabel("Degrees 0~180");
% 

% settings new vars

csv_export = [t r2 r2_new];
writematrix(csv_export,'motions.csv') 


r2 = r2_new;
r3 = r3_new;
r4 = r4_new;

sim("robot_arm_sol",1:N) 

