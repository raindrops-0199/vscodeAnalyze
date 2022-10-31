# Using NLTK and matplotlib to visualize and analyze GitHub PR and Issues  

我们以vscode为例，对其GitHub仓库的PR，Issue等进行了数据爬取，统计与建模分析，从而进行了需求的抽取与分类
 本仓库提供了我们所爬取的源数据，统计分析的源代码和最终的结果与报告。

<hr></hr>

### 文件结构

- datasource文件夹: 爬取的数据文件，包含vs code GitHub仓库的 open issue, open PR, closed PR和stack overflow上关于vs code的问题
- mask文件夹: 词云制作的mask picture
- result文件夹: 词频统计结果(csv文件)，词频统计条形图，词云图
- Report.pdf: 分析的Report
- wordFreq.ipynb: 对数据进行分析和可视化的jupyter notebook代码文件