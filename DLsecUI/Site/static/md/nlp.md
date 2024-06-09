## 可选超参数

<br>

| 超参数                 | 可选项1           | 可选项2     | 可选项3  | 
| ---------------------- | ----------------- | ----------- | -------- | 
| 数据集选择             | sst             | sst2     | ag    |                    |
| 模型结构               | bert          |     |  
| 后门攻防鲁棒性测试方法 | Lws           | Hiddenkiller       |     |


<br>

## 评估指标体系

<br>

| 指标名 | 领域                 | 说明                          |
| ------ | -------------------- | ----------------------------- |
|   𝑨𝑪𝑪     |  后门 |  模型正常任务准确率            |
|   𝑨𝑺𝑹      |  后门|  攻击的成功率                  |
|   ROBUST       |    后门鲁棒性           |  模型对非后门扰动鲁棒性        |

<br>

## 使用说明

- 这里暂时不支持自己上传模型，可以使用系统内置的BERT模型。
- 设置对应参数，可点击高级设置展开设置更多相关超参数。点击确认参数，开始运行。
- 由于存在返回值，因此需要耐心等待一段时间，评测结束后网站会自动刷新。
- 运行结束后，结果总览将呈现本次评测对应结果。

<br>

## 结果说明

<br>

结果中包含三个部分：

- 后门鲁棒性测试模块最终不同的指标和得分
- 后门指标分数雷达图可视化
- 模型最终得分