# 实验说明

## 环境依赖：
* jax
* mctx==0.0.5

## 程序调用

```
cd /projs/AE/dongshouyang/chencong/transbot/test/tvm_mcts
python tvm_mcts_basic.py
```



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

