作者：李军利	[AI-Friend](https://github.com/AI-Friend)    /  9th Jan. 2021

内容：图表示学习的损失函数；embedding的相似性度量；mrr评估方法



[TOC]

# 相似度/距离

目标顶点 $src$ 与正样本 $pos$ 或负样本 $negs$ 的相似性或者距离度量常使用 <span style='color:blue'>向量点积，余弦距离，欧式距离，pearson</span> 等方法

三种相似度/距离既有区别也有联系，通常我们把`距离度量`作为`相似度`的反义词，即距离越大，相似性越小



## dot product

点积是一个向量在另一个向量方向上的投影，物理学上，力的功就是力向量与距离向量的点积

点积/内积 有 `代数方式和几何方式` 两种定义方式
$$
\vec{a} \centerdot \vec{b} = |\vec{a}||\vec{b}| \cos \theta = \sum_{i=1}^{i=n} {a_i b_i} = a^T b = b^T a \tag{1}
$$
word2vec, deepwalk 等算法都使用了点积相似度，比如未使用负采样的 $softmax$ 损失
$$
\begin{aligned}
Loss(\theta) &= argmax(\theta) \sum_{w \epsilon Text}\sum_{C \epsilon Context(w)} log{P(c|w;\theta)} \\
&= argmax(\theta) \sum_{w \epsilon Text}\sum_{C \epsilon Context(w)} log\frac{e^{u_c  \centerdot u_w}}{\sum_{c' \epsilon corpus}e^{u_{c'}  \centerdot u_w}}
\end{aligned}
\tag{2}
$$
期望点积结果 $u_c  \centerdot u_w$ 经过$softmax$ 后尽可能地与真实分布 $[0,0,...1...,0,0]$ 相同，$1$ 是中心词索引。

优化 公式$(2)$ 的点积实际上是在优化 公式$(1)$ 中 $a,b$ 的模长和夹角，使用负采样时，正样本对的标签为1，负样本对的标签为0，那么训练的过程中，期望正样本对的点积越来越接近1，夹角越来越接近0，最终理想的情况是，在embedding空间里，正样本对的向量越来越接近，但是在训练过程中，$|a|,|b|$也是会发生变化的，模长和 $\cos \theta$ 共同决定最终的相似度，所以最终，两个相近的词或者相邻顶点的embedding向量不容易完全重合，而是相对其它不相关的词/顶点更加接近。

点积相似度对模长敏感，当我们只关注向量方向时，点积可除以模长($L2$范数)，这就是公式$(3)$

## cosine and pearson

向量内积的值域是无界限的，除以模长作归一化，归一化后的向量点积结果就是夹角余弦值
$$ {\frac{\vec{a} \centerdot \vec{b}}{|a||b|}
cos \theta = \frac{\vec{a} \centerdot \vec{b}}{|a||b|} = \frac{\sum_{i=1}^{i=n} {a_i b_i}}{\sqrt{\sum_{i=1}^{i=n} {a_i^2}}\sqrt{\sum_{i=1}^{i=n} {b_i^2}}}  \tag{3}
$$
余弦相似度只从方向上区分差异，经常用于信息检索，文本挖掘，文档相似度（TF-IDF）和图片相似性（histogram）的计算，但是它对绝对的数值（模长）不敏感（尺度不变性）， <span style='color:blue'>只能分辨个体在维之间的差异，无法衡量每个维度上数值的差异</span>，另外余弦相似度受到向量的平移影响（平移可变性），如果将 $a$ 平移到 $a+1$, 余弦值就会改变。

比如两个用户按5分制评论同一题材的两部电影，评分分别为 $(1,2)、(4,5)$，两个评分的余弦相似度是0.98，但从评分上看，第一个用户似乎不喜欢这两部电影，第二个用户则比较喜欢，余弦相似度对数值的不敏感导致了结果的误差。如何调整这种误差呢？一种方法是所有维度上的数值都减去一个均值，比如两部电影的评分均值是 $(2,3)$，那么调整后的评分是 $(-1,-1)、(2,2)$，方向完全相反，余弦相似度为-1，将所以评分作+1的平移，均值也加1，所以调整后的余弦相似度结果不变，具有平移不变性。

假设评分标准差异不大，很明显，第一个用户不喜欢该题材电影，第二个用户则喜欢。调整前的相似度反映了用户在 <span style='color:blue'>两个电影中更偏爱哪一个</span> 上的相似（维度之间的差异，）。调整后的则反映了用户在 <span style='color:blue'>喜欢的电影类型</span> 上是否一致（维度间和维度内的数值差异），这种调整的余弦相似度算法就是 $pearson \ correlation$ [1](http://brenocon.com/blog/2012/03/cosine-similarity-pearson-correlation-and-ols-coefficients/) 
$$
corr(\vec{a},\vec{b}) = \frac{\sum_{i=1}^{i=n} {(a_i-\overline{a})(b_i-\overline{b})}}{\sqrt{\sum_{i=1}^{i=n} {(a_i-\overline{a})^2}}\sqrt{\sum_{i=1}^{i=n} {(b_i-\overline{b})^2}}}  \tag{4}
$$
$pearson \ correlation$ 既具有平移不变性，又具有尺度不变性，而且当向量有缺失值时，我们一般用均值填充， $pearson \ correlation$ 可以处理缺失值的情况。

 <span style='color:blue'>评标准有横向和纵向的差距，一种是用户间的标准不同，一种是维度间标准的不同，常被混淆</span>。

*如果用户评分标准差距很大，比如第一个用户的评分标准相当高，他打2分就是很高了，那么余弦相似度更适合，但是一般情况下，评分标准差距可控。另一种情况是用户在维度之间的评分标准差距很大，比如所有用户对A题材的电影普遍评分较高，对B题材的电影普遍评分较低，或者两个题材的评分单位不一样，这是维度间的级别差距，pearson 相关系数更合适。*

## euclidean distance

欧氏距离源自欧氏空间中两点间的距离公式，是最常用的相似性度量方法，向量间的距离越小越相似，0表示两个向量完全相同，欧氏距离默认为每一个维度给予相同的权重，如果某个维度差距特别大，很容易决定整个距离，可以使用加权距离，或者标准化。
$$
d(a,b) = \sqrt{\sum_{i=1}^{i=n} {(a_i-b_i)^2}}  \tag{5}
$$
模长归一化的欧式距离
$$
d(a,b) = \sqrt{(a-b)^T (a-b)} = \sqrt{a^T a + b^T b -a^Tb -b^Ta} = \sqrt{2(1-\cos \theta)} \tag{6}
$$


## 适用情景

* 欧式距离 体现数值上的绝对差异，维度间的数值和量级同样重要，比如根据登录次数，观看次数和时长，来分析用户活跃度，$(1,10), (10,100)$  的活跃度差距很大，欧氏距离很大，表示差异很大

* 余弦距离 体现方向上的相对差异，比如统计是否观看电影 $(1,0), (0,1)$ 差异很大，不适合欧氏距离

  余弦距离不受每个用户评分标准不同和观看影片数量不一样的影响

  余弦相似度值域 $[-1,1]$ , 转换成余弦距离 $[0,1]$ 有两种方式，一般采用距离为0表示相似度最高
  $$
  \begin{cases}
  dis(a,b) = (1-cos(a,b))/2;\  0表示方向完全相同，1表示相反 \\
  dis(a,b) = (1+cos(a,b))/2;\  0表示方向完全相反，1表示相同 
  \tag{7} 
  \end{cases}
  $$

* 点积 模长归一化时，点积等于余弦距离，未归一化时，如果希望模长/幅度起作用，点积更好
* $pearson \ correlation$ 缺失值多，希望关注每个维度内的数据差异，数据受维度级别膨胀影响（不同的用户评价不同维度的标准不同），pearson相似度度量是基于一对物品的普通用户的评分与这些物品的平均评分的偏离程度

*余弦距离满足狭义距离定义的正定性和对称性，但是不满足三角不等性，机器学习中的KL距离／相对熵只满足正定性，但是不妨碍我们将它们称为距离*



# 损失函数与距离

根据 [公式2](#dot product) ，期望点积结果 $u_c  \centerdot u_w$ 经过$softmax$ 后尽可能地与真实分布 $[0,0,...1...,0,0]$ 相同，一般使用负采样减少 $softmax$ 的计算，这里我们使用 $sigmoid\  cross\ entropy$ 作二分类代替 $softmax$ 的多分类，具体是将负样本的标签设为0，正样本的标签设为1，也可以直接使用 $softmax$ 

```python
# distance 表示 向量点积，余弦距离，欧式距离，pearson 等多种距离/相似度度量
src_pos = distance(src, pos)  # 一对一
src_negs = distance(src, negs)  # 一对多
# 以dot距离为例, 欧式距离相反，pos对应zeros_like，
# 余弦距离需要依据使用公式(7)的哪个公式而定
loss_pos = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(src_pos), logits=src_pos)
loss_negs = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(src_negs), logits=src_negs)   
loss = tf.reduce_mean(loss_pos) + tf.reduce_mean(loss_negs)

# follow tensorflow
"""Computes sigmoid cross entropy given `logits`.

Measures the probability error in discrete classification tasks in which each
class is independent and not mutually exclusive.  For instance, one could
perform multilabel classification where a picture can contain both an elephant
and a dog at the same time.

For brevity, let `x = logits`, `z = labels`.  The logistic loss is

    z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
  = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
  = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
  = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
  = (1 - z) * x + log(1 + exp(-x))
  = x - x * z + log(1 + exp(-x))

For x < 0, to avoid overflow in exp(-x), we reformulate the above

    x - x * z + log(1 + exp(-x))
  = log(exp(x)) - x * z + log(1 + exp(-x))
  = - x * z + log(1 + exp(x))

Hence, to ensure stability and avoid overflow, the implementation uses this
equivalent formulation

  max(x, 0) - x * z + log(1 + exp(-abs(x)))
"""

# 根据上面的公式，以欧式距离为例，计算应的loss的值域
"""
欧式距离值域 0->正无穷: 
当x = src_pos（正样本对距离）, z = 0 时，loss_pos = x - log(1+exp(-x)) 
-> 一阶导数大于0，当欧氏距离等于0，有最小值log2 ->loss_pos >= log2

当x = src_neg（负样本对距离）, z = 1 时，loss_neg = log(1 + exp(-x))
-> 一阶导数小于0, 当欧氏距离等于正无穷，有最小值,接近0，当欧氏距离等于0，有最大值log2 
-> log2 >= loss_neg > 0

当正样本对距离等于0，负样本对距离等于正无穷时，loss取得最小值,无限接近log2
log2 < loss < 正无穷
"""
```



# Mean Reciprocal Rank

平均倒数排名（Mean Reciprocal Rank）是通用的对搜索算法进行评价的机制，即第一个结果匹配，分数为1，第二个匹配分数为0.5，第n个匹配分数为1/n，如果没有匹配的句子分数为0。最终的分数为所有得分之和。
$$
mrr = \frac 1 {|Q|} \sum_{i=1}^{|Q|} \frac {1} {rank_i} \tag{8}
$$
在 [无监督损失](#损失函数与距离) 代码里，我们期望最后训练的效果：正样本对的相似度明显高于负样本对，对相似度作排序后，就可以应用Mean Reciprocal Rank，衡量正样本对的相似度和负样本对的不相似度，之所以使用 mrr，是因为

```python
num = 1 + neg_num  # 一般地，正样本1个，负样本多个
# distance 表示 向量点积，余弦距离，欧式距离，pearson 等多种距离/相似度度量
src_pos = distance(src, pos)  # 一对一，三维[batch_size,1,embedding_dim]
src_negs = distance(src, negs)  # 一对多，二维 [batch_size,neg_num,embedding_dim]
logits = tf.concat([src_negs, src_pos], axis=-1)
# labels = tf.concat([tf.zeros_like(src_negs)， tf.ones_like(src_pos)], axis=-1)
_, indices_of_ranks = tf.nn.top_k(logits, k=size)
_, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
# 欧式距离度量
# mrr = tf.reduce_mean(tf.reciprocal(tf.to_float(size - ranks[:, -1])))
# dot度量，余弦距离需要依据使用公式(7)的哪个公式而定
mrr = tf.reduce_mean(tf.reciprocal(tf.to_float(ranks[:, -1] + 1)))

# 假设负样本个数是5，那么mrr的最大值是1，最小值是1/6
```

mrr 被广泛用在允许返回多个结果的问题，或者目前还比较难以解决的问题中（由于如果只返回top1的结果，精确率或召回率会很差，所以在技术不成熟的情况下，先返回多个结果）。在这类问题中，系统会对每一个返回的结果给一个置信度（打分），然后根据置信度排序，将得分高的结果排在前面返回。

核心思想很简单：返回的结果集的优劣，跟第一个正确答案的位置有关，第一个正确答案越靠前，结果越好。

比如，推荐系统向3个用户推荐5个产品，第一个用户购买了第一个，第二个用户购买了第3个，第三个用户没有购买这5个产品，那么推荐系统的 mrr 得分
$$
mrr = (1+1/3+0)/3 = 4/9
$$


# References

[1] [Cosine similarity, Pearson correlation, and OLS coefficients](http://brenocon.com/blog/2012/03/cosine-similarity-pearson-correlation-and-ols-coefficients/)