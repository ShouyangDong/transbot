from func_test import run_compilation, runtime_test, unit_test, perf_test

def reward_function(y, c):
    """
    This function evaluates the output of a program and returns a reward based on the following criteria:
    R1 if the program cannot be compiled (y != c)
    R2 if the program has a runtime error, timeout, or failed any unit test
    R3 if the program passed all unit tests
    R4 if the program passed all unit tests and improved runtime

    Parameters:
    y (str): The actual output of the program.
    c (str): The expected output of the program.

    Returns:
    int: The reward value based on the program's performance.
    """
    if not run_compilation()：
        return -1

    elif run_compilation() and not runtime_test():
        return 0

    elif runtime_test() and not unit_test():
        return 1

    elif unit_test() and not perf_test():
        return 2

    else:
        return 5
