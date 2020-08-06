# wangjue
这是一个基于Transformer的聊天机器人模型

--dataset 选择自己的训练数据集
    数据集需要满足以下格式
    E
    M 。。。。。
    M 。。。。。

    E
    M 。。。。
    M 。。。。。
    。。。。。。
--if_train 选择是否训练自己的模型,默认 True
--if_cuda 选择是否选择gpu,默认 False
--h  选择多少个注意力头,默认8
--d_model  选择embedding_dim,默认256
--batch_size   选择批次大小,默认128
--dropout 选择dropout的大小,默认0.5
--test_file 如果选择测试模型,请输入你的模型参数文件
--lr  选择你的学习率,默认0.005
--layers 选择你的神经网络深度,默认16
