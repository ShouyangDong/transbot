from tvm_environments import build_env, ActionSpace


def _main():
    env = build_env()

    optimizer_ids = [2, 2, 3, 3, 0]
    optimizer_ops = [ActionSpace[_ii] for _ii in optimizer_ids]

    for _ii in range(100):
        stat, reward = env.perform_action(optimizer_ops)
        print(f"reward_{_ii}", reward)
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

    optimizer_ids = [
        2,
        2,
        3,
        3,
    ]
    optimizer_ops = [ActionSpace[_ii] for _ii in optimizer_ids]
    stat, reward = env.perform_action(optimizer_ops)
    print("reward2", reward)


if __name__ == "__main__":
    _main()
