from func_test import check_compilation, check_runtime, run_tests


def reward_function(file_path, test_script_path="test.py"):
    """
    This function evaluates the output of a program and returns a reward based on the following criteria:
    R1 if the program cannot be compiled (y != c)
    R2 if the program has a runtime error, timeout, or failed any unit test
    R3 if the program passed all unit tests
    R4 if the program passed all unit tests and improved runtime

    Parameters:
    file_path (str): The tested program path.
    test_script_path (str): The unit test file path.

    Returns:
    int: The reward value based on the program's performance.
    """
    # First compile the code
    reward = check_compilation(file_path.replace(".cu", ".so"), file_path)
    if reward > 0:
        runtime_reward = check_runtime(file_path, test_script_path)
        reward += runtime_reward

    # Run unit test
    test_reward = run_tests(file_path, test_script_path)
    reward += test_reward

    # TODO: add peft test
    # # Run perf test
    # perf_reward = check_peft(test_script_path)
    # reward += perf_reward
    return reward
