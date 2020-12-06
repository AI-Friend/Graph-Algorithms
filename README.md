作者：李军利	[AI-Friend](https://github.com/AI-Friend)    /  14th June 2020

内容：图基础和图引擎；**图算法**：`图挖掘`、 `图表示学习` 、`图神经网络`、 `知识表示学习/知识图谱三元组` 

（`Graph Mining` 、`Graph Embedding`、`Graph Neural Network`、`Knowledge-Graph Embedding`）

编程相关：`Linux`、`C++`、`Python`、`TensorFlow`、`Pytorch`、`DGL`、`PyG`、`networkx`、`HDFS`

写作动力：随着图引擎和图算法研究的深入，涉及越来越广，希望在 [Graph-Algorithms](https://github.com/AI-Friend/Graph-Algorithms) 里记录一些总结和思考

*git 显示md文档的公式和图片太费劲了，clone下来或者看pdf更易读*

分类：旨在获取embedding的无监督算法称为 `图表示学习` ;  `GNN`常常是监督学习; 知识图谱相关的称为 `KG-Embedding`（*<span style='color:green;background:white;'>
  我的分类很主观，基于游走的算法常称为图表示算法，基于邻居汇聚的叫 GNN</span>*）

| 分类   | 笔记           | 论文                                                         | 代码                                                | 异构 | 属性 |
| ------ | -------------- | ------------------------------------------------------------ | --------------------------------------------------- | ---- | ---- |
| 基础   | [Graph Theory] |                                                              |                                                     |      |      |
| 基础   | [Gemini]       | [A Computation-Centric Distributed Graph ···](https://www.usenix.org/system/files/conference/osdi16/osdi16-zhu.pdf) |                                                     |      |      |
| 基础   | [信息与熵]     |                                                              |                                                     |      |      |
| 基础   | [Alias method] |                                                              |                                                     |      |      |
|        |                |                                                              |                                                     |      |      |
| 图表示 | [deepwalk]     | [Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf) | [master](https://github.com/phanein/deepwalk)       | 0    | 0    |
| 图表示 | [node2vec ]    | [Scalable Feature Learning for Networks](https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf) | [master](https://github.com/aditya-grover/node2vec) | 0    | 0    |
| 图表示 | [复现node2vec] |                                                              |                                                     |      |      |
| 图表示 | [LINE ]        | [Large-scale Information Network Embedding](https://arxiv.org/abs/1503.03578) | [master](https://github.com/tangjianpku/LINE)       | 0    | 0    |
| 图表示 | [metapath2vec] | [Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) |                                                     | 1    | 0    |
| 图表示 | [DGI]          |                                                              |                                                     | 0    | 1    |
|        |                |                                                              |                                                     |      |      |
| GNN    | [GCN]          | [Semi-Supervised Classification with GCN](https://arxiv.org/abs/1609.02907) | [master](https://github.com/tkipf/gcn)              | 0    | 1    |
| GNN    | [GraphSage]    | [Inductive Representation Learning on L-Graphs](https://arxiv.org/abs/1706.02216) | [master](https://github.com/search?q=graphsage)     | 0    | 1    |
| GNN    | [GAT]          | [Graph Attention Network](https://arxiv.org/abs/1710.10903)  | [master](https://github.com/PetarV-/GAT)            | 0    | 1    |
| GNN    | [Deep GCN]     |                                                              |                                                     | 0    | 1    |
| GNN    | [HGT]          |                                                              |                                                     | 1    | 1    |
|        |                |                                                              |                                                     |      |      |
| KG-E   | [TransE]       |                                                              |                                                     |      |      |
| KG-E   | [TransH]       |                                                              |                                                     |      |      |
| KG-E   | [TransR]       |                                                              |                                                     |      |      |
| KG-E   | [TransD]       |                                                              |                                                     |      |      |
|        |                |                                                              |                                                     |      |      |
| 图挖掘 | [(W)CC]        |                                                              |                                                     |      |      |
| 图挖掘 | [LPA]          |                                                              |                                                     |      |      |
| 图挖掘 | [SSSP & APSP]  |                                                              |                                                     |      |      |
| 图挖掘 | [InfoMap]      |                                                              |                                                     |      |      |
| 图挖掘 | [Louvain]      |                                                              |                                                     |      |      |


