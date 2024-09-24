# transbot
Transcompile code by using llm and rl.

# Install 
Install is similar to tvm. First, fill in USE_CUDA and USE_LLVM in cmake/config.cmake, like this:

```
set(USE_LLVM "/path/to/llvm-config --link-static")
set(HIDE_PRIVATE_SYMBOLS ON)
set(USE_CUDA /usr/local/cuda)
```

Then build tvm

```
mkdir -p build && cd build && cp ../cmake/config.cmake . && cmake .. && make -j && cd -
export PYTHONPATH="$PYTHONPATH:$PWD/python"
# some python package required by tvm
pip install torch attrs cloudpickle decorator psutil synr tornado xgboost
```

# Language reference
Still in progress.

See test cases.
