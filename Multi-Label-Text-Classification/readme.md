多标签分类

数据格式
  两个文件夹
  
  1.训练集   内部为json文件，格式案例为{ "question" : "IE浏览器出问题", "tags" : [ "浏览器", "操作系统" ] }
  
  2.验证集   同上
  
  预训练词向量
    下载地址： https://github.com/Embedding/Chinese-Word-Vectors


运行方式：

1. 修改数据集和词向量地址
2. run data_utils
3. run create_record
4. run train_cnn.py
