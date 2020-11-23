作者：李军利	[AI-Friend](https://github.com/AI-Friend)    /  14th June 2020

内容：图基础和图引擎；**图算法**：`图挖掘`、 `图表示学习` 、`图神经网络`、 `知识表示学习/知识图谱三元组` ;

​            `Graph Mining` 、`Graph Embedding`、`Graph Neural Network`、`Knowledge-Graph Embedding`；

计算机：`Linux`、`C++`、`Python`、`HDFS`、`TensorFlow`、`Pytorch`、`DGI`、`PyG`、`networkx`

写作动力：

随着图引擎和图算法的研究越来越深入，涉及面越来越广，希望在 [Graph-Algorithms](https://github.com/AI-Friend/Graph-Algorithms) 里分类记录一些总结/思考



| 分类   | 总结                                                         | References                                                   | 源码                                                | 异构 | 属性 |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------- | ---- | ---- |
| 基础   | [Graph Theory](https://github.com/AI-Friend/Graph-Algorithms/blob/master/%E5%9B%BE%E6%8C%96%E6%8E%98%E7%AE%97%E6%B3%95/Graph%20Theory.md) |                                                              |                                                     |      |      |
| 基础   | [Gemini]                                                     | [A Computation-Centric Distributed Graph ···](https://www.usenix.org/system/files/conference/osdi16/osdi16-zhu.pdf) |                                                     |      |      |
| 基础   | [信息与熵](https://github.com/AI-Friend/Graph-Algorithms/blob/master/%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/%E4%BF%A1%E6%81%AF%E4%B8%8E%E7%86%B5.md) |                                                              |                                                     |      |      |
|        |                                                              |                                                              |                                                     |      |      |
| 图表示 | [deepwalk](https://github.com/AI-Friend/Graph-Algorithms/blob/master/%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/deep%20walk%20and%20node2vec.md) | [Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf) | [master](https://github.com/phanein/deepwalk)       | 0    | 0    |
| 图表示 | [node2vec ](https://github.com/AI-Friend/Graph-Algorithms/blob/master/%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/deep%20walk%20and%20node2vec.md) | [Scalable Feature Learning for Networks](https://cs.stanford.edu/people/jure/pubs/node2vec-kdd16.pdf) | [master](https://github.com/aditya-grover/node2vec) | 0    | 0    |
| 图表示 | [LINE ](https://github.com/AI-Friend/Graph-Algorithms/blob/master/%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/LINE.md) | [Large-scale Information Network Embedding](https://arxiv.org/abs/1503.03578) | [master](https://github.com/tangjianpku/LINE)       | 0    | 0    |
| 图表示 | [metapath2vec](https://github.com/AI-Friend/Graph-Algorithms/blob/master/%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/MetaPath2Vec.md) | [Learning for Heterogeneous Networks](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf) |                                                     | 1    | 0    |
| 图表示 | [Un-GraphSage](https://github.com/AI-Friend/Graph-Algorithms/blob/master/%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/GraphSage.md) | [Inductive Representation Learning on L-Graphs](https://arxiv.org/abs/1706.02216) | [master](https://github.com/search?q=graphsage)     | 0    | 1    |
| 图表示 | DGI                                                          |                                                              |                                                     | 0    | 1    |
|        |                                                              |                                                              |                                                     |      |      |
| GNN    | [GCN](https://github.com/AI-Friend/Graph-Algorithms/blob/master/%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/GCN.md) | [Semi-Supervised Classification with GCN](https://arxiv.org/abs/1609.02907) | [master](https://github.com/tkipf/gcn)              | 0    | 1    |
| GNN    | [GraphSage](https://github.com/AI-Friend/Graph-Algorithms/blob/master/%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/GraphSage.md) | [Inductive Representation Learning on L-Graphs](https://arxiv.org/abs/1706.02216) | [master](https://github.com/search?q=graphsage)     | 0    | 1    |
| GNN    | [GAT](https://github.com/AI-Friend/Graph-Algorithms/blob/master/%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/GAT.md) | [Graph Attention Network](https://arxiv.org/abs/1710.10903)  | [master](https://github.com/PetarV-/GAT)            | 0    | 1    |
| GNN    | Deep GCN                                                     |                                                              |                                                     | 0    | 1    |
| GNN    | HGT                                                          |                                                              |                                                     | 1    | 1    |
|        |                                                              |                                                              |                                                     |      |      |
| KG-E   |                                                              |                                                              |                                                     |      |      |
|        |                                                              |                                                              |                                                     |      |      |
| 图挖掘 | (W)CC                                                        |                                                              |                                                     |      |      |
| 图挖掘 | LPA                                                          |                                                              |                                                     |      |      |
| 图挖掘 | SSSP & APSP                                                  |                                                              |                                                     |      |      |
| 图挖掘 | InfoMap                                                      |                                                              |                                                     |      |      |
| 图挖掘 | Louvain                                                      |                                                              |                                                     |      |      |



