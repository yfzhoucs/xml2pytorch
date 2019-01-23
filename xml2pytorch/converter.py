import collections
import torch.nn as nn
import torch.nn.functional as F
import torch
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np

def main():
    simple_cnn = convertXML("./xml_examples/simpleCNN.xml")
    print(simple_cnn)    
    x = torch.randn(1, 3, 32, 32)
    y = simple_cnn(x)
    print(y)


def convertXML(xml_filename):
    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(xml_filename)
    graph = DOMTree.documentElement
     
    nets = graph.getElementsByTagName("net")
    net = nets[0]
    net_model = Net(net)

    return net_model


class Net(nn.Module):
    def __init__(self, net_xmlNode):
        super(Net, self).__init__()
        self.net_xmlNode = net_xmlNode
        self.layers = []

        layers_xmlNode = net_xmlNode.getElementsByTagName("layer")

        for layer_xmlNode in layers_xmlNode: 
            net_style = None
            args_dict = None
            activation = None

            # 去除 xml 里的换行符
            for i in range(len(layer_xmlNode.childNodes)-1, -1, -1):
                if(layer_xmlNode.childNodes[i].nodeName == "#text"):
                    layer_xmlNode.removeChild(layer_xmlNode.childNodes[i])

            # 隐藏层类型
            net_styles = layer_xmlNode.getElementsByTagName("net_style")
            if len(net_styles) > 0:
                net_style = net_styles[0].childNodes[0].data

                # 隐藏层的默认参数
                args_dict = self.__default_args_for_layer(net_style)
                # 隐藏层，xml 追加的参数
                args_dict = self.__check_args_from_xml(args_dict, layer_xmlNode)

            # 判断激活函数
            activations = layer_xmlNode.getElementsByTagName("activation")
            if len(activations) > 0:
                activation = activations[0].childNodes[0].data
                

            # 声明网络和激活函数
            if net_style is not None:
                self.__append_layer(net_style, args_dict)
            if activation is not None:
                self.__append_activation(activation)

    
    def __append_activation(self, activation):
        if activation == "relu":
            self.layers.append(F.relu)
        elif activation == "elu":
            self.layers.append(F.elu)
        elif activation == "selu":
            self.layers.append(F.selu)
        elif activation == "celu":
            self.layers.append(F.celu)
        elif activation == "leaky_relu":
            self.layers.append(F.leaky_relu)
        elif activation == "logsigmoid":
            self.layers.append(F.logsigmoid)
        elif activation == "softmax":
            self.layers.append(F.softmax)
        elif activation == "tanh":
            self.layers.append(torch.tanh)
        elif activation == "sigmoid":
            self.layers.append(torch.sigmoid)


    def __append_layer(self, net_style, args_dict):
        args_values_list = list(args_dict.values())
        if net_style == "Conv2d":
            self.layers.append(nn.Conv2d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4],
                args_values_list[5],
                args_values_list[6],
                args_values_list[7]))
        elif net_style == "MaxPool2d":
            self.layers.append(nn.MaxPool2d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4],
                args_values_list[5]))
        elif net_style == "Linear":
            self.layers.append(nn.Linear(args_values_list[0],
                args_values_list[1],
                args_values_list[2]))   
        elif net_style == "reshape":
            # 如果是特殊情况 reshape，就直接将目标向量尺寸传入
            # print(type(args_values_list[0]))
            self.layers.append(args_values_list[0])
        elif net_style == "Conv1d":
            self.layers.append(nn.Conv1d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4],
                args_values_list[5],
                args_values_list[6],
                args_values_list[7]))   
        elif net_style == "Conv3d":
            self.layers.append(nn.Conv3d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4],
                args_values_list[5],
                args_values_list[6],
                args_values_list[7]))   
        elif net_style == "ConvTranspose1d":
            self.layers.append(nn.ConvTranspose1d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4],
                args_values_list[5],
                args_values_list[6],
                args_values_list[7],
                args_values_list[8]))   
        elif net_style == "ConvTranspose2d":
            self.layers.append(nn.ConvTranspose2d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4],
                args_values_list[5],
                args_values_list[6],
                args_values_list[7],
                args_values_list[8]))   
        elif net_style == "ConvTranspose3d":
            self.layers.append(nn.ConvTranspose3d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4],
                args_values_list[5],
                args_values_list[6],
                args_values_list[7],
                args_values_list[8]))   
        elif net_style == "Unfold":
            self.layers.append(nn.Unfold(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3]))   
        elif net_style == "Fold":
            self.layers.append(nn.Unfold(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4]))
        elif net_style == "MaxPool1d":
            self.layers.append(nn.MaxPool1d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4],
                args_values_list[5]))
        elif net_style == "MaxPool3d":
            self.layers.append(nn.MaxPool3d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4],
                args_values_list[5]))
        elif net_style == "MaxUnpool1d":
            self.layers.append(nn.MaxUnpool1d(args_values_list[0],
                args_values_list[1],
                args_values_list[2]))
        elif net_style == "MaxUnpool2d":
            self.layers.append(nn.MaxUnpool2d(args_values_list[0],
                args_values_list[1],
                args_values_list[2]))
        elif net_style == "MaxUnpool3d":
            self.layers.append(nn.MaxUnpool3d(args_values_list[0],
                args_values_list[1],
                args_values_list[2]))
        elif net_style == "AvgPool1d":
            self.layers.append(nn.AvgPool1d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4]))
        elif net_style == "AvgPool2d":
            self.layers.append(nn.AvgPool2d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4]))
        elif net_style == "AvgPool3d":
            self.layers.append(nn.AvgPool3d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4]))
        elif net_style == "FractionalMaxPool2d":
            self.layers.append(nn.FractionalMaxPool2d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4]))
        elif net_style == "LPPool1d":
            self.layers.append(nn.LPPool1d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3]))
        elif net_style == "LPPool2d":
            self.layers.append(nn.LPPool2d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3]))
        elif net_style == "AdaptiveMaxPool1d":
            self.layers.append(nn.AdaptiveMaxPool1d(args_values_list[0],
                args_values_list[1]))
        elif net_style == "AdaptiveMaxPool2d":
            self.layers.append(nn.AdaptiveMaxPool2d(args_values_list[0],
                args_values_list[1]))
        elif net_style == "AdaptiveMaxPool3d":
            self.layers.append(nn.AdaptiveMaxPool3d(args_values_list[0],
                args_values_list[1]))
        elif net_style == "AdaptiveAvgPool1d":
            self.layers.append(nn.AdaptiveAvgPool1d(args_values_list[0]))
        elif net_style == "AdaptiveAvgPool2d":
            self.layers.append(nn.AdaptiveAvgPool2d(args_values_list[0]))
        elif net_style == "AdaptiveAvgPool3d":
            self.layers.append(nn.AdaptiveAvgPool3d(args_values_list[0]))
        elif net_style == "ReflectionPad1d":
            self.layers.append(nn.ReflectionPad1d(args_values_list[0]))
        elif net_style == "ReflectionPad2d":
            self.layers.append(nn.ReflectionPad2d(args_values_list[0]))
        elif net_style == "ReplicationPad1d":
            self.layers.append(nn.ReplicationPad1d(args_values_list[0]))
        elif net_style == "ReplicationPad2d":
            self.layers.append(nn.ReplicationPad2d(args_values_list[0]))
        elif net_style == "ReplicationPad3d":
            self.layers.append(nn.ReplicationPad3d(args_values_list[0]))
        elif net_style == "ZeroPad2d":
            self.layers.append(nn.ZeroPad2d(args_values_list[0]))
        elif net_style == "ConstantPad1d":
            self.layers.append(nn.ConstantPad1d(args_values_list[0],
                args_values_list[1]))
        elif net_style == "ConstantPad2d":
            self.layers.append(nn.ConstantPad2d(args_values_list[0],
                args_values_list[1]))
        elif net_style == "ConstantPad3d":
            self.layers.append(nn.ConstantPad3d(args_values_list[0],
                args_values_list[1]))
        elif net_style == "ELU":
            self.layers.append(nn.ELU(args_values_list[0],
                args_values_list[1]))
        elif net_style == "Hardshrink":
            self.layers.append(nn.Hardshrink(args_values_list[0]))            
        elif net_style == "Hardtanh":
            self.layers.append(nn.Hardtanh(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4]))
        elif net_style == "LeakyReLU":
            self.layers.append(nn.LeakyReLU(args_values_list[0],
                args_values_list[1]))
        elif net_style == "LogSigmoid":
            self.layers.append(nn.LogSigmoid())
        elif net_style == "PReLU":
            self.layers.append(nn.PReLU(args_values_list[0],
                args_values_list[1]))
        elif net_style == "ReLU":
            self.layers.append(nn.ReLU(args_values_list[0]))
        elif net_style == "ReLU6":
            self.layers.append(nn.ReLU6(args_values_list[0]))
        elif net_style == "RReLU":
            self.layers.append(nn.RReLU(args_values_list[0],
                args_values_list[1],
                args_values_list[2]))
        elif net_style == "SELU":
            self.layers.append(nn.SELU(args_values_list[0]))
        elif net_style == "CELU":
            self.layers.append(nn.CELU(args_values_list[0],
                args_values_list[1]))
        elif net_style == "Sigmoid":
            self.layers.append(nn.Sigmoid())
        elif net_style == "Softplus":
            self.layers.append(nn.Softplus(args_values_list[0],
                args_values_list[1]))
        elif net_style == "Softshrink":
            self.layers.append(nn.Softshrink(args_values_list[0]))
        elif net_style == "Softsign":
            self.layers.append(nn.Softsign())
        elif net_style == "Tanh":
            self.layers.append(nn.Tanh())
        elif net_style == "Tanhshrink":
            self.layers.append(nn.Tanhshrink())
        elif net_style == "Threshold":
            self.layers.append(nn.Threshold(args_values_list[0],
                args_values_list[1],
                args_values_list[2]))
        elif net_style == "Softmin":
            self.layers.append(nn.Softmin(args_values_list[0]))
        elif net_style == "Softmax":
            self.layers.append(nn.Softmax(args_values_list[0]))
        elif net_style == "Softmax2d":
            self.layers.append(nn.Softmax2d())
        elif net_style == "LogSoftmax":
            self.layers.append(nn.LogSoftmax(args_values_list[0]))
        elif net_style == "AdaptiveLogSoftmaxWithLoss":
            self.layers.append(nn.AdaptiveLogSoftmaxWithLoss(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4]))
        elif net_style == "BatchNorm1d":
            self.layers.append(nn.BatchNorm1d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4]))
        elif net_style == "BatchNorm2d":
            self.layers.append(nn.BatchNorm2d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4]))
        elif net_style == "BatchNorm3d":
            self.layers.append(nn.BatchNorm3d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4]))
        elif net_style == "GroupNorm":
            self.layers.append(nn.GroupNorm(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3]))
        elif net_style == "InstanceNorm1d":
            self.layers.append(nn.InstanceNorm1d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4]))
        elif net_style == "InstanceNorm2d":
            self.layers.append(nn.InstanceNorm2d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4]))
        elif net_style == "InstanceNorm3d":
            self.layers.append(nn.InstanceNorm3d(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3],
                args_values_list[4]))
        elif net_style == "LayerNorm":
            self.layers.append(nn.LayerNorm(args_values_list[0],
                args_values_list[1],
                args_values_list[2]))
        elif net_style == "LocalResponseNorm":
            self.layers.append(nn.LocalResponseNorm(args_values_list[0],
                args_values_list[1],
                args_values_list[2],
                args_values_list[3]))
        elif net_style == "Linear":
            self.layers.append(nn.Linear(args_values_list[0],
                args_values_list[1],
                args_values_list[2]))
        elif net_style == "Dropout":
            self.layers.append(nn.Dropout(args_values_list[0],
                args_values_list[1]))
        elif net_style == "Dropout2d":
            self.layers.append(nn.Dropout2d(args_values_list[0],
                args_values_list[1]))
        elif net_style == "Dropout3d":
            self.layers.append(nn.Dropout3d(args_values_list[0],
                args_values_list[1]))
        elif net_style == "AlphaDropout":
            self.layers.append(nn.AlphaDropout(args_values_list[0],
                args_values_list[1]))


    # 从 xml 中读取参数并覆盖默认参数
    def __check_args_from_xml(self, args_dict, layer_xml_node):
        args_keys = list(args_dict.keys())
        for args_key in args_keys:
            args_value_xml_node_list = layer_xml_node.getElementsByTagName(args_key)
            if len(args_value_xml_node_list) > 0:
                args_value_string = args_value_xml_node_list[0].childNodes[0].data
                args_value = eval(args_value_string)
                args_dict[args_key] = args_value
        return args_dict


    # 网络的默认参数
    def __default_args_for_layer(self, net_style):
        args_dict = collections.OrderedDict()
        if net_style == "Conv2d":
            args_dict["in_channels"] = None
            args_dict["out_channels"] = None
            args_dict["kernel_size"] = None
            args_dict["stride"] = 1
            args_dict["padding"] = 0
            args_dict["dilation"] = 1
            args_dict["groups"] = 1
            args_dict["bias"] = True
        elif net_style == "MaxPool2d":
            args_dict["kernel_size"] = None
            args_dict["stride"] = None
            args_dict["padding"] = 0
            args_dict["dilation"] = 1
            args_dict["return_indices"] = False
            args_dict["ceil_mode"] = False
        elif net_style == "Linear":
            args_dict["in_features"] = None
            args_dict["out_features"] = None
            args_dict["bias"] = True
        elif net_style == "reshape":
            args_dict["dimensions"] = None
        elif net_style == "Conv1d":
            args_dict["in_channels"] = None
            args_dict["out_channels"] = None
            args_dict["kernel_size"] = None
            args_dict["stride"] = 1
            args_dict["padding"] = 0
            args_dict["dilation"] = 1
            args_dict["groups"] = 1
            args_dict["bias"] = True
        elif net_style == "Conv3d":
            args_dict["in_channels"] = None
            args_dict["out_channels"] = None
            args_dict["kernel_size"] = None
            args_dict["stride"] = 1
            args_dict["padding"] = 0
            args_dict["dilation"] = 1
            args_dict["groups"] = 1
            args_dict["bias"] = True
        elif net_style == "ConvTranspose1d":
            args_dict["in_channels"] = None
            args_dict["out_channels"] = None
            args_dict["kernel_size"] = None
            args_dict["stride"] = 1
            args_dict["padding"] = 0
            args_dict["output_padding"] = 0
            args_dict["groups"] = 1
            args_dict["bias"] = True
            args_dict["dilation"] = 1
        elif net_style == "ConvTranspose2d":
            args_dict["in_channels"] = None
            args_dict["out_channels"] = None
            args_dict["kernel_size"] = None
            args_dict["stride"] = 1
            args_dict["padding"] = 0
            args_dict["output_padding"] = 0
            args_dict["groups"] = 1
            args_dict["bias"] = True
            args_dict["dilation"] = 1
        elif net_style == "ConvTranspose3d":
            args_dict["in_channels"] = None
            args_dict["out_channels"] = None
            args_dict["kernel_size"] = None
            args_dict["stride"] = 1
            args_dict["padding"] = 0
            args_dict["output_padding"] = 0
            args_dict["groups"] = 1
            args_dict["bias"] = True
            args_dict["dilation"] = 1
        elif net_style == "Unfold":
            args_dict["kernel_size"] = None
            args_dict["dilation"] = 1
            args_dict["padding"] = 0
            args_dict["stride"] = 1
        elif net_style == "Fold":
            args_dict["output_size"] = None
            args_dict["kernel_size"] = None
            args_dict["dilation"] = 1
            args_dict["padding"] = 0
            args_dict["stride"] = 1
        elif net_style == "MaxPool1d":
            args_dict["kernel_size"] = None
            args_dict["stride"] = None
            args_dict["padding"] = 0
            args_dict["dilation"] = 1
            args_dict["return_indices"] = False
            args_dict["ceil_mode"] = False
        elif net_style == "MaxPool3d":
            args_dict["kernel_size"] = None
            args_dict["stride"] = None
            args_dict["padding"] = 0
            args_dict["dilation"] = 1
            args_dict["return_indices"] = False
            args_dict["ceil_mode"] = False
        elif net_style == "MaxUnpool1d":
            args_dict["kernel_size"] = None
            args_dict["stride"] = None
            args_dict["padding"] = 0
        elif net_style == "MaxUnpool2d":
            args_dict["kernel_size"] = None
            args_dict["stride"] = None
            args_dict["padding"] = 0
        elif net_style == "MaxUnpool3d":
            args_dict["kernel_size"] = None
            args_dict["stride"] = None
            args_dict["padding"] = 0
        elif net_style == "AvgPool1d":
            args_dict["kernel_size"] = None
            args_dict["stride"] = None
            args_dict["padding"] = 0
            args_dict["ceil_mode"] = False
            args_dict["count_include_pad"] = True
        elif net_style == "AvgPool2d":
            args_dict["kernel_size"] = None
            args_dict["stride"] = None
            args_dict["padding"] = 0
            args_dict["ceil_mode"] = False
            args_dict["count_include_pad"] = True
        elif net_style == "AvgPool3d":
            args_dict["kernel_size"] = None
            args_dict["stride"] = None
            args_dict["padding"] = 0
            args_dict["ceil_mode"] = False
            args_dict["count_include_pad"] = True
        elif net_style == "FractionalMaxPool2d":
            args_dict["kernel_size"] = None
            args_dict["output_size"] = None
            args_dict["output_ratio"] = None
            args_dict["return_indices"] = False
            args_dict["_random_samples"] = None
        elif net_style == "LPPool1d":
            args_dict["norm_type"] = None
            args_dict["kernel_size"] = None
            args_dict["stride"] = None
            args_dict["ceil_mode"] = False
        elif net_style == "LPPool2d":
            args_dict["norm_type"] = None
            args_dict["kernel_size"] = None
            args_dict["stride"] = None
            args_dict["ceil_mode"] = False
        elif net_style == "AdaptiveMaxPool1d":
            args_dict["output_size"] = None
            args_dict["return_indices"] = False
        elif net_style == "AdaptiveMaxPool2d":
            args_dict["output_size"] = None
            args_dict["return_indices"] = False
        elif net_style == "AdaptiveMaxPool3d":
            args_dict["output_size"] = None
            args_dict["return_indices"] = False
        elif net_style == "AdaptiveAvgPool1d":
            args_dict["output_size"] = None
        elif net_style == "AdaptiveAvgPool2d":
            args_dict["output_size"] = None
        elif net_style == "AdaptiveAvgPool3d":
            args_dict["output_size"] = None
        elif net_style == "ReflectionPad1d":
            args_dict["padding"] = None
        elif net_style == "ReflectionPad2d":
            args_dict["padding"] = None
        elif net_style == "ReplicationPad1d":
            args_dict["padding"] = None
        elif net_style == "ReplicationPad2d":
            args_dict["padding"] = None
        elif net_style == "ReplicationPad3d":
            args_dict["padding"] = None
        elif net_style == "ZeroPad2d":
            args_dict["padding"] = None
        elif net_style == "ConstantPad1d":
            args_dict["padding"] = None
            args_dict["value"] = None
        elif net_style == "ConstantPad2d":
            args_dict["padding"] = None
            args_dict["value"] = None
        elif net_style == "ConstantPad3d":
            args_dict["padding"] = None
            args_dict["value"] = None
        elif net_style == "ELU":
            args_dict["alpha"] = 1.0
            args_dict["inplace"] = False
        elif net_style == "Hardshrink":
            args_dict["lambd"] = 0.5
        elif net_style == "Hardtanh":
            args_dict["min_val"] = -1.0
            args_dict["max_val"] = 1.0
            args_dict["inplace"] = False
            args_dict["min_value"] = None
            args_dict["max_value"] = None
        elif net_style == "LeakyReLU":
            args_dict["negative_slope"] = 0.01
            args_dict["inplace"] = False
        elif net_style == "LogSigmoid":
            args_dict = None
        elif net_style == "PReLU":
            args_dict["num_parameters"] = 1
            args_dict["init"] = 0.25
        elif net_style == "ReLU":
            args_dict["inplace"] = False
        elif net_style == "ReLU6":
            args_dict["inplace"] = False
        elif net_style == "RReLU":
            args_dict["lower"] = 0.125
            args_dict["upper"] = 1.0 / 3.0
            args_dict["inplace"] = False
        elif net_style == "SELU":
            args_dict["inplace"] = False
        elif net_style == "CELU":
            args_dict["alpha"] = 1.0
            args_dict["inplace"] = False
        elif net_style == "Sigmoid":
            args_dict = None
        elif net_style == "Softplus":
            args_dict["beta"] = 1
            args_dict["threshold"] = 20
        elif net_style == "Softshrink":
            args_dict["lambd"] = 0.5
        elif net_style == "Softsign":
            args_dict = None
        elif net_style == "Tanh":
            args_dict = None
        elif net_style == "Tanhshrink":
            args_dict = None
        elif net_style == "Threshold":
            args_dict["threshold"] = None
            args_dict["value"] = None
            args_dict["inplace"] = False
        elif net_style == "Softmin":
            args_dict["dim"] = None
        elif net_style == "Softmax":
            args_dict["dim"] = None
        elif net_style == "Softmax2d":
            args_dict = None
        elif net_style == "LogSoftmax":
            args_dict["dim"] = None
        elif net_style == "AdaptiveLogSoftmaxWithLoss":
            args_dict["in_features"] = None
            args_dict["n_classes"] = None
            args_dict["cutoffs"] = None
            args_dict["div_value"] = 4.0
            args_dict["head_bias"] = False
        elif net_style == "BatchNorm1d":
            args_dict["num_features"] = None
            args_dict["eps"] = 1e-05
            args_dict["momentum"] = 0.1
            args_dict["affine"] = True
            args_dict["track_running_stats"] = True
        elif net_style == "BatchNorm2d":
            args_dict["num_features"] = None
            args_dict["eps"] = 1e-05
            args_dict["momentum"] = 0.1
            args_dict["affine"] = True
            args_dict["track_running_stats"] = True
        elif net_style == "BatchNorm3d":
            args_dict["num_features"] = None
            args_dict["eps"] = 1e-05
            args_dict["momentum"] = 0.1
            args_dict["affine"] = True
            args_dict["track_running_stats"] = True
        elif net_style == "GroupNorm":
            args_dict["num_groups"] = None
            args_dict["num_channels"] = None
            args_dict["eps"] = 1e-05
            args_dict["affine"] = True
        elif net_style == "InstanceNorm1d":
            args_dict["num_features"] = None
            args_dict["eps"] = 1e-05
            args_dict["momentum"] = 0.1
            args_dict["affine"] = True
            args_dict["track_running_stats"] = False
        elif net_style == "InstanceNorm2d":
            args_dict["num_features"] = None
            args_dict["eps"] = 1e-05
            args_dict["momentum"] = 0.1
            args_dict["affine"] = True
            args_dict["track_running_stats"] = False
        elif net_style == "InstanceNorm3d":
            args_dict["num_features"] = None
            args_dict["eps"] = 1e-05
            args_dict["momentum"] = 0.1
            args_dict["affine"] = True
            args_dict["track_running_stats"] = False
        elif net_style == "LayerNorm":
            args_dict["normalized_shape"] = None
            args_dict["eps"] = 1e-05
            args_dict["elementwise_affine"] = True
        elif net_style == "LocalResponseNorm":
            args_dict["size"] = None
            args_dict["alpha"] = 0.0001
            args_dict["beta"] = 0.75
            args_dict["k"] = 1.0
        elif net_style == "Linear":
            args_dict["in_features"] = None
            args_dict["out_features"] = None
            args_dict["bias"] = True
        elif net_style == "Dropout":
            args_dict["p"] = 0.5
            args_dict["inplace"] = False
        elif net_style == "Dropout2d":
            args_dict["p"] = 0.5
            args_dict["inplace"] = False
        elif net_style == "Dropout3d":
            args_dict["p"] = 0.5
            args_dict["inplace"] = False
        elif net_style == "AlphaDropout":
            args_dict["p"] = 0.5
            args_dict["inplace"] = False
        # elif net_style == "RNNCell":
        #     args_dict["input_size"] = None
        #     args_dict["hidden_size"] = None
        #     args_dict["bias"] = True
        #     args_dict["nonlinearity"] = 'tanh'

        return args_dict

    def forward(self, x):
        for i in range(len(self.layers)):
            # 是.view，转换 Tensor 的形状
            if isinstance(self.layers[i], list):
                # print(x.shape)
                # print(self.layers[i])
                x = x.view(self.layers[i])
            # 是一个隐藏层
            else:
                # print(x.shape)
                x = self.layers[i](x)
        return x



class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.view([-1, 16 * 5 * 5])
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = self.fc3(x)
        return x

if __name__ == "__main__": 
    main()