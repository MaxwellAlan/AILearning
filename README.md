# AILearning （Geekhoo 组织）

`简介：专注于AI领域的资源共享和整合`

* **Geekhoo - 学习机器学习群【629470233】<a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=30e5f1123a79867570f665aa3a483ca404b1c3f77737bc01ec520ed5f078ddef"><img border="0" src="images/MainPage/Geekhoo-group.png" alt="Geekhoo - 学习机器学习群[629470233]" title="Geekhoo - 学习机器学习群[629470233]"></a>**
* [**【干货】机器学习项目流程**](docs/main/机器学习项目流程.md)
* [推荐：你可能会用到的计算机书籍](docs/main/你可能会用到的计算机书籍.md)
* AI = Python + MachineLearning（Sklearn） + DeepLearning（PyTorch）

> 编程入门学习建议

```
具体的做法：

1、选择一个方向：产品、实施、测试、运维、编程、架构、算法、人工智能。
2、选择一门编程语言：Shell、JAVA、Python、C、C++、C#、PHP、GO
3、然后计算机基础：可以看看《计算机导论》和《计算机操作系统》，《计算机网络》《XXX编程语言教学指南》
4、然后开始着手你想要做的项目：Web前端? Web前端？自动化运维？数据分析？数据挖掘？
5、然后就开始去横冲直撞的找工作和选择Offer

说完这些干货后，我作为过来人想告诉你：
    书还是别看了，网上去搜索《某某技术》的视频教程？？
    视频网站：百度搜索？网易云课堂？淘宝？慕课网？51CTO？哔哩哔哩？传智播客？

最后送你一句话：加油！！  路可远可近，但愿遇见都是好人一枚！
```

> 项目开发流程

```
1.理解实际问题，抽象为机器学习能处理的数学问题

    理解实际业务场景问题是机器学习的第一步，机器学习中特征工程和模型训练都是非常费时的，深入理解要处理的问题，能避免走很多弯路。理解问题，包括明确可以获得的数据，机器学习的目标是分类、回归还是聚类。如果都不是的话，考虑将它们转变为机器学习问题。参考[机器学习分类](http://www.cnblogs.com/wxquare/p/5281753.html)能帮助从问题提炼出一个合适的机器学习方法。

2.获取数据

    获取数据包括获取原始数据以及从原始数据中经过特征工程从原始数据中提取训练、测试数据。机器学习比赛中原始数据都是直接提供的，但是实际问题需要自己获得原始数据。“ 数据决定机器学习结果的上限，而算法只是尽可能的逼近这个上限”，可见数据在机器学习中的作用。总的来说数据要有具有“代表性”，对于分类问题，数据偏斜不能过于严重，不同类别的数据数量不要有数个数量级的差距。不仅如此还要对评估数据的量级，样本数量、特征数量，估算训练模型对内存的消耗。如果数据量太大可以考虑减少训练样本、降维或者使用分布式机器学习系统。

3.特征工程

    特征工程是非常能体现一个机器学习者的功底的。特征工程包括从原始数据中特征构建、特征提取、特征选择，非常有讲究。深入理解实际业务场景下的问题，丰富的机器学习经验能帮助我们更好的处理特征工程。特征工程做的好能发挥原始数据的最大效力，往往能够使得算法的效果和性能得到显著的提升，有时能使简单的模型的效果比复杂的模型效果好。数据挖掘的大部分时间就花在特征工程上面，是机器学习非常基础而又必备的步骤。数据预处理、数据清洗、筛选显著特征、摒弃非显著特征等等都非常重要，建议深入学习。

4.模型训练、诊断、调优

    现在有很多的机器学习算法的工具包，例如sklearn，使用非常方便，真正考验水平的根据对算法的理解调节参数，使模型达到最优。当然，能自己实现算法的是最牛的。模型诊断中至关重要的是判断过拟合、欠拟合，常见的方法是绘制学习曲线，交叉验证。通过增加训练的数据量、降低模型复杂度来降低过拟合的风险，提高特征的数量和质量、增加模型复杂来防止欠拟合。诊断后的模型需要进行进一步调优，调优后的新模型需要重新诊断，这是一个反复迭代不断逼近的过程，需要不断的尝试，进而达到最优的状态。

5.模型验证、误差分析

    模型验证和误差分析也是机器学习中非常重要的一步，通过测试数据，验证模型的有效性，观察误差样本，分析误差产生的原因，往往能使得我们找到提升算法性能的突破点。误差分析主要是分析出误差来源与数据、特征、算法。

6.模型融合

    一般来说实际中，成熟的机器算法也就那么些，提升算法的准确度主要方法是模型的前端（特征工程、清洗、预处理、采样）和后端的模型融合。在机器学习比赛中模型融合非常常见，基本都能使得效果有一定的提升。这篇[博客](http://www.cnblogs.com/wxquare/p/5440664.html)中提到了模型融合的方法，主要包括一人一票的统一融合，线性融合和堆融合。

7.上线运行 

    这一部分内容主要跟工程实现的相关性比较大。工程上是结果导向，模型在线上运行的效果直接决定模型的成败。 不单纯包括其准确程度、误差等情况，还包括其运行的速度(时间复杂度)、资源消耗程度（空间复杂度）、稳定性是否可接受。这些工作流程主要是工程实践上总结出的一些经验。并不是每个项目都包含完整的一个流程。这里的部分只是一个指导性的说明，只有大家自己多实践，多积累项目经验，才会有自己更深刻的认识。
```

## [**初级-Python**](初级-Python)

面向0基础编程的同学，请仔细Coding

- [01. **Python 工具**](初级-Python/01-python-tools)
     - [01.01 Python 简介](初级-Python/01-python-tools/01.01-python-overview.ipynb)
     - [01.02 Ipython 解释器](初级-Python/01-python-tools/01.02-ipython-interpreter.ipynb)
     - [01.03 Ipython notebook](初级-Python/01-python-tools/01.03-ipython-notebook.ipynb)
     - [01.04 使用 Anaconda](初级-Python/01-python-tools/01.04-use-anaconda.ipynb)
- [02. **Python 基础**](初级-Python/02-python-essentials)
     - [02.01 Python 入门演示](初级-Python/02-python-essentials/02.01-a-tour-of-python.ipynb)
     - [02.02 Python 数据类型](初级-Python/02-python-essentials/02.02-python-data-types.ipynb)
     - [02.03 数字](初级-Python/02-python-essentials/02.03-numbers.ipynb)
     - [02.04 字符串](初级-Python/02-python-essentials/02.04-strings.ipynb)
     - [02.05 索引和分片](初级-Python/02-python-essentials/02.05-indexing-and-slicing.ipynb)
     - [02.06 列表](初级-Python/02-python-essentials/02.06-lists.ipynb)
     - [02.07 可变和不可变类型](初级-Python/02-python-essentials/02.07-mutable-and-immutable-data-types.ipynb)
     - [02.08 元组](初级-Python/02-python-essentials/02.08-tuples.ipynb)
     - [02.09 列表与元组的速度比较](初级-Python/02-python-essentials/02.09-speed-comparison-between-list-&-tuple.ipynb)
     - [02.10 字典](初级-Python/02-python-essentials/02.10-dictionaries.ipynb)
     - [02.11 集合](初级-Python/02-python-essentials/02.11-sets.ipynb)
     - [02.12 不可变集合](初级-Python/02-python-essentials/02.12-frozen-sets.ipynb)
     - [02.13 Python 赋值机制](初级-Python/02-python-essentials/02.13-how-python-assignment-works.ipynb)
     - [02.14 判断语句](初级-Python/02-python-essentials/02.14-if-statement.ipynb)
     - [02.15 循环](初级-Python/02-python-essentials/02.15-loops.ipynb)
     - [02.16 列表推导式](初级-Python/02-python-essentials/02.16-list-comprehension.ipynb)
     - [02.17 函数](初级-Python/02-python-essentials/02.17-functions.ipynb)
     - [02.18 模块和包](初级-Python/02-python-essentials/02.18-modules-and-packages.ipynb)
     - [02.19 异常](初级-Python/02-python-essentials/02.19-exceptions.ipynb)
     - [02.20 警告](初级-Python/02-python-essentials/02.20-warnings.ipynb)
     - [02.21 文件读写](初级-Python/02-python-essentials/02.21-file-IO.ipynb)
- [03. **Numpy**](初级-Python/03-numpy)
     - [03.01 Numpy 简介](初级-Python/03-numpy/03.01-numpy-overview.ipynb)
     - [03.02 Matplotlib 基础](初级-Python/03-numpy/03.02-matplotlib-basics.ipynb)
     - [03.03 Numpy 数组及其索引](初级-Python/03-numpy/03.03-numpy-arrays.ipynb)
     - [03.04 数组类型](初级-Python/03-numpy/03.04-array-types.ipynb)
     - [03.05 数组方法](初级-Python/03-numpy/03.05-array-calculation-method.ipynb)
     - [03.06 数组排序](初级-Python/03-numpy/03.06-sorting-numpy-arrays.ipynb)
     - [03.07 数组形状](初级-Python/03-numpy/03.07-array-shapes.ipynb)
     - [03.08 对角线](初级-Python/03-numpy/03.08-diagonals.ipynb)
     - [03.09 数组与字符串的转换](初级-Python/03-numpy/03.09-data-to-&-from-string.ipynb)
     - [03.10 数组属性方法总结](初级-Python/03-numpy/03.10-array-attribute-&-method-overview-.ipynb)
     - [03.11 生成数组的函数](初级-Python/03-numpy/03.11-array-creation-functions.ipynb)
     - [03.12 矩阵](初级-Python/03-numpy/03.12-matrix-object.ipynb)
     - [03.13 一般函数](初级-Python/03-numpy/03.13-general-functions.ipynb)
     - [03.14 向量化函数](初级-Python/03-numpy/03.14-vectorizing-functions.ipynb)
     - [03.15 二元运算](初级-Python/03-numpy/03.15-binary-operators.ipynb)
     - [03.16 ufunc 对象](初级-Python/03-numpy/03.16-universal-functions.ipynb)
     - [03.17 choose 函数实现条件筛选](初级-Python/03-numpy/03.17-choose.ipynb)
     - [03.18 数组广播机制](初级-Python/03-numpy/03.18-array-broadcasting.ipynb)
     - [03.19 数组读写](初级-Python/03-numpy/03.19-reading-and-writing-arrays.ipynb)
     - [03.20 结构化数组](初级-Python/03-numpy/03.20-structured-arrays.ipynb)
     - [03.21 记录数组](初级-Python/03-numpy/03.21-record-arrays.ipynb)
     - [03.22 内存映射](初级-Python/03-numpy/03.22-memory-maps.ipynb)
     - [03.23 从 Matlab 到 Numpy](初级-Python/03-numpy/03.23-from-matlab-to-numpy.ipynb)
- [04. **Scipy**](初级-Python/04-scipy)
     - [04.01 SCIentific PYthon 简介](初级-Python/04-scipy/04.01-scienticfic-python-overview.ipynb)
     - [04.02 插值](初级-Python/04-scipy/04.02-interpolation-with-scipy.ipynb)
     - [04.03 概率统计方法](初级-Python/04-scipy/04.03-statistics-with-scipy.ipynb)
     - [04.04 曲线拟合](初级-Python/04-scipy/04.04-curve-fitting.ipynb)
     - [04.05 最小化函数](初级-Python/04-scipy/04.05-minimization-in-python.ipynb)
     - [04.06 积分](初级-Python/04-scipy/04.06-integration-in-python.ipynb)
     - [04.07 解微分方程](初级-Python/04-scipy/04.07-ODEs.ipynb)
     - [04.08 稀疏矩阵](初级-Python/04-scipy/04.08-sparse-matrix.ipynb)
     - [04.09 线性代数](初级-Python/04-scipy/04.09-linear-algbra.ipynb)
     - [04.10 稀疏矩阵的线性代数](初级-Python/04-scipy/04.10-sparse-linear-algebra.ipynb)
- [05. **Python 进阶**](初级-Python/05-advanced-python)
     - [05.01 sys 模块简介](初级-Python/05-advanced-python/05.01-overview-of-the-sys-module.ipynb)
     - [05.02 与操作系统进行交互：os 模块](初级-Python/05-advanced-python/05.02-interacting-with-the-OS---os.ipynb)
     - [05.03 CSV 文件和 csv 模块](初级-Python/05-advanced-python/05.03-comma-separated-values.ipynb)
     - [05.04 正则表达式和 re 模块](初级-Python/05-advanced-python/05.04-regular-expression.ipynb)
     - [05.05 datetime 模块](初级-Python/05-advanced-python/05.05-datetime.ipynb)
     - [05.06 SQL 数据库](初级-Python/05-advanced-python/05.06-sql-databases.ipynb)
     - [05.07 对象关系映射](初级-Python/05-advanced-python/05.07-object-relational-mappers.ipynb)
     - [05.08 函数进阶：参数传递，高阶函数，lambda 匿名函数，global 变量，递归](初级-Python/05-advanced-python/05.08-functions.ipynb)
     - [05.09 迭代器](初级-Python/05-advanced-python/05.09-iterators.ipynb)
     - [05.10 生成器](初级-Python/05-advanced-python/05.10-generators.ipynb)
     - [05.11 with 语句和上下文管理器](初级-Python/05-advanced-python/05.11-context-managers-and-the-with-statement.ipynb)
     - [05.12 修饰符](初级-Python/05-advanced-python/05.12-decorators.ipynb)
     - [05.13 修饰符的使用](初级-Python/05-advanced-python/05.13-decorator-usage.ipynb)
     - [05.14 operator, functools, itertools, toolz, fn, funcy 模块](初级-Python/05-advanced-python/05.14-the-operator-functools-itertools-toolz-fn-funcy-module.ipynb)
     - [05.15 作用域](初级-Python/05-advanced-python/05.15-scope.ipynb)
     - [05.16 动态编译](初级-Python/05-advanced-python/05.16-dynamic-code-execution.ipynb)
- [06. **Matplotlib**](初级-Python/06-matplotlib)
     - [06.01 Pyplot 教程](初级-Python/06-matplotlib/06.01-pyplot-tutorial.ipynb)
     - [06.02 使用 style 来配置 pyplot 风格](初级-Python/06-matplotlib/06.02-customizing-plots-with-style-sheets.ipynb)
     - [06.03 处理文本（基础）](初级-Python/06-matplotlib/06.03-working-with-text---basic.ipynb)
     - [06.04 处理文本（数学表达式）](初级-Python/06-matplotlib/06.04-working-with-text---math-expression.ipynb)
     - [06.05 图像基础](初级-Python/06-matplotlib/06.05-image-tutorial.ipynb)
     - [06.06 注释](初级-Python/06-matplotlib/06.06-annotating-axes.ipynb)
     - [06.07 标签](初级-Python/06-matplotlib/06.07-legend.ipynb)
     - [06.08 figures, subplots, axes 和 ticks 对象](初级-Python/06-matplotlib/06.08-figures,-subplots,-axes-and-ticks.ipynb)
     - [06.09 不要迷信默认设置](初级-Python/06-matplotlib/06.09-do-not-trust-the-defaults.ipynb)
     - [06.10 各种绘图实例](初级-Python/06-matplotlib/06.10-different-plots.ipynb)
- [07. **使用其他语言进行扩展**](初级-Python/07-interfacing-with-other-languages)
     - [07.01 简介](初级-Python/07-interfacing-with-other-languages/07.01-introduction.ipynb)
     - [07.02 Python 扩展模块](初级-Python/07-interfacing-with-other-languages/07.02-python-extension-modules.ipynb)
     - [07.03 Cython：Cython 基础，将源代码转换成扩展模块](初级-Python/07-interfacing-with-other-languages/07.03-cython-part-1.ipynb)
     - [07.04 Cython：Cython 语法，调用其他C库](初级-Python/07-interfacing-with-other-languages/07.04-cython-part-2.ipynb)
     - [07.05 Cython：class 和 cdef class，使用 C++](初级-Python/07-interfacing-with-other-languages/07.05-cython-part-3.ipynb)
     - [07.06 Cython：Typed memoryviews](初级-Python/07-interfacing-with-other-languages/07.06-cython-part-4.ipynb)
     - [07.07 生成编译注释](初级-Python/07-interfacing-with-other-languages/07.07-profiling-with-annotations.ipynb)
     - [07.08 ctypes](初级-Python/07-interfacing-with-other-languages/07.08-ctypes.ipynb)
- [08. **面向对象编程**](初级-Python/08-object-oriented-programming)
     - [08.01 简介](初级-Python/08-object-oriented-programming/08.01-oop-introduction.ipynb)
     - [08.02 使用 OOP 对森林火灾建模](初级-Python/08-object-oriented-programming/08.02-using-oop-model-a-forest-fire.ipynb)
     - [08.03 什么是对象？](初级-Python/08-object-oriented-programming/08.03-what-is-a-object.ipynb)
     - [08.04 定义 class](初级-Python/08-object-oriented-programming/08.04-writing-classes.ipynb)
     - [08.05 特殊方法](初级-Python/08-object-oriented-programming/08.05-special-method.ipynb)
     - [08.06 属性](初级-Python/08-object-oriented-programming/08.06-properties.ipynb)
     - [08.07 森林火灾模拟](初级-Python/08-object-oriented-programming/08.07-forest-fire-simulation.ipynb)
     - [08.08 继承](初级-Python/08-object-oriented-programming/08.08-inheritance.ipynb)
     - [08.09 super() 函数](初级-Python/08-object-oriented-programming/08.09-super.ipynb)
     - [08.10 重定义森林火灾模拟](初级-Python/08-object-oriented-programming/08.10-refactoring-the-forest-fire-simutation.ipynb)
     - [08.11 接口](初级-Python/08-object-oriented-programming/08.11-interfaces.ipynb)
     - [08.12 共有，私有和特殊方法和属性](初级-Python/08-object-oriented-programming/08.12-public-private-special-in-python.ipynb)
     - [08.13 多重继承](初级-Python/08-object-oriented-programming/08.13-multiple-inheritance.ipynb)
- [09. **Theano 基础**](初级-Python/09-theano)
     - [09.01 Theano 简介及其安装](初级-Python/09-theano/09.01-introduction-and-installation.ipynb)
     - [09.02 Theano 基础](初级-Python/09-theano/09.02-theano-basics.ipynb)
     - [09.03 Theano 在 Windows 上的配置](初级-Python/09-theano/09.03-gpu-on-windows.ipynb)
     - [09.04 Theano 符号图结构](初级-Python/09-theano/09.04-graph-structures.ipynb)
     - [09.05 Theano 配置和编译模式](初级-Python/09-theano/09.05-configuration-settings-and-compiling-modes.ipynb)
     - [09.06 Theano 条件语句](初级-Python/09-theano/09.06-conditions-in-theano.ipynb)
     - [09.07 Theano 循环：scan（详解）](初级-Python/09-theano/09.07-loop-with-scan.ipynb)
     - [09.08 Theano 实例：线性回归](初级-Python/09-theano/09.08-linear-regression.ipynb)
     - [09.09 Theano 实例：Logistic 回归](初级-Python/09-theano/09.09-logistic-regression-.ipynb)
     - [09.10 Theano 实例：Softmax 回归](初级-Python/09-theano/09.10-softmax-on-mnist.ipynb)
     - [09.11 Theano 实例：人工神经网络](初级-Python/09-theano/09.11-net-on-mnist.ipynb)
     - [09.12 Theano 随机数流变量](初级-Python/09-theano/09.12-random-streams.ipynb)
     - [09.13 Theano 实例：更复杂的网络](初级-Python/09-theano/09.13-modern-net-on-mnist.ipynb)
     - [09.14 Theano 实例：卷积神经网络](初级-Python/09-theano/09.14-convolutional-net-on-mnist.ipynb)
     - [09.15 Theano tensor 模块：基础](初级-Python/09-theano/09.15-tensor-basics.ipynb)
     - [09.16 Theano tensor 模块：索引](初级-Python/09-theano/09.16-tensor-indexing.ipynb)
     - [09.17 Theano tensor 模块：操作符和逐元素操作](初级-Python/09-theano/09.17-tensor-operator-and-elementwise-operations.ipynb)
     - [09.18 Theano tensor 模块：nnet 子模块](初级-Python/09-theano/09.18-tensor-nnet-.ipynb)
     - [09.19 Theano tensor 模块：conv 子模块](初级-Python/09-theano/09.19-tensor-conv.ipynb)
- [10. **有趣的第三方模块**](初级-Python/10-something-interesting)
     - [10.01 使用 basemap 画地图](初级-Python/10-something-interesting/10.01-maps-using-basemap.ipynb)
     - [10.02 使用 cartopy 画地图](初级-Python/10-something-interesting/10.02-maps-using-cartopy.ipynb)
     - [10.03 探索 NBA 数据](初级-Python/10-something-interesting/10.03-nba-data.ipynb)
     - [10.04 金庸的武侠世界](初级-Python/10-something-interesting/10.04-louis-cha's-kungfu-world.ipynb)
- [11. **有用的工具**](初级-Python/11-useful-tools)
     - [11.01 pprint 模块：打印 Python 对象](初级-Python/11-useful-tools/11.01-pprint.ipynb)
     - [11.02 pickle, cPickle 模块：序列化 Python 对象](初级-Python/11-useful-tools/11.02-pickle-and-cPickle.ipynb)
     - [11.03 json 模块：处理 JSON 数据](初级-Python/11-useful-tools/11.03-json.ipynb)
     - [11.04 glob 模块：文件模式匹配](初级-Python/11-useful-tools/11.04-glob.ipynb)
     - [11.05 shutil 模块：高级文件操作](初级-Python/11-useful-tools/11.05-shutil.ipynb)
     - [11.06 gzip, zipfile, tarfile 模块：处理压缩文件](初级-Python/11-useful-tools/11.06-gzip,-zipfile,-tarfile.ipynb)
     - [11.07 logging 模块：记录日志](初级-Python/11-useful-tools/11.07-logging.ipynb)
     - [11.08 string 模块：字符串处理](初级-Python/11-useful-tools/11.08-string.ipynb)
     - [11.09 collections 模块：更多数据结构](初级-Python/11-useful-tools/11.09-collections.ipynb)
     - [11.10 requests 模块：HTTP for Human](初级-Python/11-useful-tools/11.10-requests.ipynb)
- [12. **Pandas**](初级-Python/12-pandas)
     - [12.01 十分钟上手 Pandas](初级-Python/12-pandas/12.01-ten-minutes-to-pandas.ipynb)
     - [12.02 一维数据结构：Series](初级-Python/12-pandas/12.02-series-in-pandas.ipynb)
     - [12.03 二维数据结构：DataFrame](初级-Python/12-pandas/12.03-dataframe-in-pandas.ipynb)

## [**中级-MachineLearning**](中级-MachineLearning<Sklearn>)

面向基础编程转型做 MachineLearning，请仔细Coding

| 第一部分-分类 | 第二部分 回归 | 第三部分 无监督学习 | 第四部分 其他工具 |
| --- | --- | --- | --- |
| 1」 [机器学习基础](中级-MachineLearning/docs/1.机器学习基础.md) | 8」 [预测数值型数据：回归](中级-MachineLearning/docs/8.预测数值型数据：回归.md) | 10」 [k-means聚类](中级-MachineLearning/docs/10.k-means聚类.md) | 13」 [利用PCA来简化数据](中级-MachineLearning/docs/13.利用PCA来简化数据.md) |
| 2」 [k-近邻算法](中级-MachineLearning/docs/2.k-近邻算法.md) | 9」 [树回归](中级-MachineLearning/docs/9.树回归.md) | 11」 [使用Apriori算法进行关联分析](中级-MachineLearning/docs/11.使用Apriori算法进行关联分析.md) | 14」 [利用SVD简化数据](中级-MachineLearning/docs/14.利用SVD简化数据.md) | 
| 3」 [决策树](中级-MachineLearning/docs/3.决策树.md) | | 12」 [使用FP-growth算法发现频繁项集](中级-MachineLearning/docs/12.使用FP-growth算法来高效发现频繁项集.md) | 15」 [大数据与MapReduce](中级-MachineLearning/docs/15.大数据与MapReduce.md) | 
| 4」 [朴素贝叶斯](中级-MachineLearning/docs/4.朴素贝叶斯.md)
| 5」 [Logistic回归](中级-MachineLearning/docs/5.Logistic回归.md)
| 6」 [支持向量机](中级-MachineLearning/docs/6.支持向量机.md)
| 7」 [集成方法-随机森林、AdaBoost](中级-MachineLearning/docs/7.集成方法-随机森林和AdaBoost.md)


## [**高级-PyTorch**](高级-DeepLearning<PyTorch>)

面向 MachineLearning 转型做 DeepLearning，请仔细Coding

[代码地址：高级-PyTorch/tutorial-contents](高级-PyTorch/tutorial-contents)

> PyTorch 简介

* [1.1 Why?](http://www.pytorchtutorial.com/1-1-why-pytorch/)
* [1.2 安装](http://www.pytorchtutorial.com/1-2-install-pytorch/)

> PyTorch 神经网络基础

* [2.1 Torch 或 Numpy](http://www.pytorchtutorial.com/2-1-torch-vs-numpy/)
* [2.2 变量 (Variable)](http://www.pytorchtutorial.com/2-2-variable/)
* [2.3 激励函数 (Activation)](http://www.pytorchtutorial.com/2-3-activation/)

> 建造第一个神经网络

* [3.1 关系拟合 (回归 Regression)](http://www.pytorchtutorial.com/3-1-regression/)
* [3.2 区分类型 (分类 Classification)](http://www.pytorchtutorial.com/3-2-classification/)
* [3.3 快速搭建回归神经网络](http://www.pytorchtutorial.com/3-3-sequential-cnn-model/)
* [3.4 保存和恢复模型](http://www.pytorchtutorial.com/3-4-save-and-restore-model/)
* [3.5 数据读取 (Data Loader)](http://www.pytorchtutorial.com/3-5-data-loader/)
* [3.6 Optimizer 优化器](http://www.pytorchtutorial.com/3-6-optimizer/)

> 高级神经网络结构

* [4.1 CNN 卷积神经网络](http://www.pytorchtutorial.com/4-1-cnn/)
* [4.2 RNN 循环神经网络 (分类 Classification)](http://www.pytorchtutorial.com/4-2-rnn-for-classification/)
* [4.3 RNN 循环神经网络 (回归 Regression)](http://www.pytorchtutorial.com/4-3-rnn-for-regression/)
* [4.4 AutoEncoder (自编码/非监督学习)](http://www.pytorchtutorial.com/4-4-autoencoder/)
* [4.5 DQN 强化学习 (Reinforcement Learning)](http://www.pytorchtutorial.com/4-5-dqn-reinforcement-learning/)
* [4.6 GAN (Generative Adversarial Nets 生成对抗网络)](http://www.pytorchtutorial.com/4-6-gan-generative-adversarial-nets/)

> 高阶内容

* [5.1 为什么 Torch 是动态的](http://www.pytorchtutorial.com/5-1-why-dynamic/)
* [5.2 GPU 加速运算](http://www.pytorchtutorial.com/5-2-gpu-in-pytorch/)
* [5.3 Dropout 防止过拟合](http://www.pytorchtutorial.com/5-3-dropout-to-prevent-overfitting/)
* [5.4 Batch Normalization 批标准化](http://www.pytorchtutorial.com/5-4-batch-normalization/)

* [查看此套教程完整目录](http://www.pytorchtutorial.com/mofan-pytorch-tutorials-list/)

## [Kaggle 项目比赛]((docs/4-Kaggle))

| 项目名称 | 文档地址 | 代码地址 | 作者信息 | 更新时间 |
| :--: | :--: | :--: | :--: | :--: |
| 手写数字识别 | [文档]() | [代码](src/4-Kaggle/digit-recognizer/svm-python3.6.py) | [片刻](https://github.com/jiangzhonglian) | 2017-12-16 |
| 房屋预测 | [文档](docs/4-Kaggle/house-prices.md) | [代码](src/4-Kaggle/house-prices/tmp.py) | [忧郁一休](http://blog.csdn.net/youyuyixiu/article/details/72841703) | 2017-12-16 |
