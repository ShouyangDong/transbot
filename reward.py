from func_test import check_compilation, check_runtime, run_tests, check_perf

def reward_function(code, test_script_path="test.cu"):
    """
    This function evaluates the output of a program and returns a reward based on the following criteria:
    R1 if the program cannot be compiled (y != c)
    R2 if the program has a runtime error, timeout, or failed any unit test
    R3 if the program passed all unit tests
    R4 if the program passed all unit tests and improved runtime

    Parameters:
    code (str): The tested program.

    Returns:
    int: The reward value based on the program's performance.
    """
    # First compile the code
    reward = check_compilation(code, test_script_path)

    if reward > 0:
        runtime_reward = check_runtime("output")
        reward += runtime_reward

    # Run unit test
    test_reward = run_tests(test_script_path)
    reward += test_reward

    # Run perf test
    perf_reward = check_peft(test_script_path)
    reward += perf_reward
    return reward
