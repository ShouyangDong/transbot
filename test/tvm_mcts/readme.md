# 实验说明

<<<<<<< HEAD
conda activate py39

=======
>>>>>>> 80a7971a960a6f3e8c51208ce1d1fba49f137455
## 环境依赖：
* jax
* mctx==0.0.5

## 程序调用

```
cd /projs/AE/dongshouyang/chencong/transbot/test/tvm_mcts
python tvm_mcts_basic.py
```


<<<<<<< HEAD
## 代码介绍

### tvm_environments.py
* TvmGo 是环境类，实现了reset和step函数。
* perform_action使用守杨原算法。增加多补编译平均时间方案。

### tvm mcts basic 算法说明
* 算法基于deepmind mctx库
* 算法基于随机gumbel policy方案。见论文：https://openreview.net/forum?id=bERaNdoegnO
### tvm mcts basic 超参数说明：
* FLAGS.num_simulations 采样个数，个数越多计算耗时越长，也更大的概率找到最优
* env.optimizer_len, 优化组合的长度，也就是说假设action长度是A，则优化的空间为 A**env.optimizer_len
### mcts的关键设置
* mctx.RootFnOutput： 设置根节点的先验，根节点和非根节点采用的action策略不同。
* mctx.RecurrentFnOutput: 设置非根节点的先验和超参数。
* mctx.gumbel_muzero_policy：设置搜索策略


### 输出结果
* terminal 下输出最优组合的id和reward
* 输出优化路径图: ./tvm_search_tree.png


=======
>>>>>>> 80a7971a960a6f3e8c51208ce1d1fba49f137455

## 实验分析
* 当前机器并非独占，因此时间度量可能存在问题。
* 实验见tvm_result_check.py


实验 optimizer_ids = [2,2,3,3,0]
进行100次编译和时间统计。最高的reward为22，最低为3。如下所示：

# 100 轮测试，最高reward22，最低3
# 可能需要独占机器，才可以获得正确的值。
# reward_0 12.213831508166656
# reward_1 13.008130081300813
# reward_2 10.81081081081081
# reward_3 12.8
# reward_4 7.655502392344497
# reward_5 3.2
# reward_6 6.557403293910468
# reward_7 3.2258128028383375
# reward_8 6.20157386133517
# reward_9 11.940298507462686
# reward2 0.002859358755465498

