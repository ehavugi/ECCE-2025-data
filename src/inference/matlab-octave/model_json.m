str1 = "models/Model";
str2 = "E";
filename=strcat(str1,str2,".json")
%filename="ModelN87.json"
text=fileread(filename);
object = jsondecode(text);

relu = @(x) max(0.0, x);
T=[[0],[0],[0],[1]];
B=[0.1*sin(linspace(-pi/2,2*pi-pi/2,24))];
f=[[200e3]];
input_value=cat(2,log10(f),log10(abs(B)),T);
input_value=reshape(input_value,[29,1]);
% Layer 1
L1=object.layers_0_weight;
L1_bias=object.layers_0_bias;
L1_out=relu(L1* +L1_bias);

L2=object.layers_2_weight;
L2_bias=object.layers_2_bias;
L2_out=relu(L2*L1_out+L2_bias)

L3=object.layers_4_weight;
L3_bias=object.layers_4_bias;
L3_out=relu(L3*L2_out+L3_bias)

L4=object.layers_6_weight;
L4_bias=object.layers_6_bias;
L4_out=L4*L3_out+L4_bias
pv=10^L4_out