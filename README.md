# xml2pytorch
Using xml to define pytorch neural networks
## What can it Do
With xml2pytorch, you can easily define neural networks in xml, and then declare them in pytorch.
## Installation
### Environment
OS independent. Python3. (Not tested on Python2, but it should work.)
### Install requirements
torch>=0.4.1
numpy>=1.15.1
### Installing by pip3
pip3 install xml2pytorch
## Quick Start
### How to declare the CNN defined by a xml file
```
import torch
import xml2pytorch as xm

# declare the net defined in .xml
net = xm.convertXML(xml_filename)    

# input a random tensor
x = torch.randn(1, 3, 32, 32)
y = net(x)
print(y)
```
### How to define a simple CNN in xml
```
<graph>
	<net>
		<layer>
			<net_style>Conv2d</net_style>
			<in_channels>3</in_channels>
			<out_channels>6</out_channels>
			<kernel_size>5</kernel_size>
		</layer>	
		<layer>
			<net_style>ELU</net_style>
		</layer>	
		<layer>
			<net_style>MaxPool2d</net_style>
			<kernel_size>2</kernel_size>
			<stride>2</stride>
			<activation>logsigmoid</activation>
		</layer>
		<layer>
			<net_style>Conv2d</net_style>
			<in_channels>6</in_channels>
			<out_channels>16</out_channels>
			<kernel_size>5</kernel_size>
			<activation>relu</activation>
		</layer>	
		<layer>
			<net_style>MaxPool2d</net_style>
			<kernel_size>2</kernel_size>
			<stride>2</stride>
			<activation>relu</activation>
		</layer>
		<layer>
			<net_style>reshape</net_style>
			<dimensions>[-1, 16*5*5]</dimensions>
		</layer>
		<layer>
			<net_style>Linear</net_style>
			<in_features>400</in_features> 
			<out_features>120</out_features>
			<activation>tanh</activation>
		</layer>
		<layer>
			<net_style>Linear</net_style>
			<in_features>120</in_features> 
			<out_features>84</out_features>
			<activation>sigmoid</activation>
		</layer>
		<layer>
			<net_style>Linear</net_style>
			<in_features>84</in_features>
			<out_features>10</out_features>
			<activation>softmax</activation>
		</layer>
	</net>
</graph>
```
