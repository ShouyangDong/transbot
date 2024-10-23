/projs/AE/dongshouyang/chencong/transbot/experments.md
# 实验说明

## 安装llm
https://tvm.apache.org/docs/install/from_source.html
https://cn.linux-console.net/?p=15141

#--
https://tvm.apache.org/docs/install/from_source.html

export TVM_LIBRARY_PATH=/home/tinavi/Project/shouyang/tvm/build

pip install -e /projs/AE/dongshouyang/tvm/python

# 测试如下代码：

## test_schedule:
* test_tvm_auto_bind.py ✅
* test_tvm_auto_cache_read.py ✅
* test_tvm_auto_cache_write.py ✅
* test_tvm_auto_compute_location.py ✅
* test_tvm_auto_inline.py ✅
* test_tvm_auto_tensorize.py ✅
* test_tvm_search_strategy.py ✅
* test_tvm_tune_add.py ❌
* test_tvm_tune_matmul.py ✅
* test_tvm_tune_tir.py ✅

## tuning
* evolutionary_search.py  ✅
* mcts.py
* test_mcts.py


## 优化说明
程序的优化入口应是：evolutionary_search.py。 目标使用RL替换原进化算法。
问题： 为什么使用进化算法解决该问题，原代码应该使用的是xgboost


### 问题：
1） 为什么不同的action 会得到相同的state。见 states = [perform_action(mod, target, action) for action in population]
2） ranks = argsort(argsort(scores))  的作用是什么？？
3） 为什么会 scores[i] < best_eval，score 从object 出，最高分，应该是最好的。



# 运行说明
conda activate py39
export TVM_LIBRARY_PATH=/projs/AE/dongshouyang/tvm/build


