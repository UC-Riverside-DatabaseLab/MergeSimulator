#!/usr/bin/python

from MergePolicy import BigtablePolicy, BinomialPolicy, ExploringPolicy, MinLatencyPolicy, ConstantPolicy
import matplotlib.pyplot as plt

policies = {"constant": "Constant",
            "binomial": "Binomial",
            "exploring": "Exploring",
            "google-default": "Bigtable",
            "min-latency": "MinLatency"}

flush_size = 4*(1024**2)  # 4 MB uniform flush
max_flushes = 3000


def simulate(ax, total, policy, color):
    flushes = []
    sum_size = 0
    cost = []
    for i in range(total):
        f, start, old_comps, new_comps = policy.flush()
        flushes.append(policy.flush_count())
        sum_size += (f + sum(new_comps))
        cost.append(sum_size / (1024 ** 3))  # In GB
    line, = ax.plot(flushes, cost, color=color, lw=2.5)
    print("Done {0}".format(policy.policy_name()))
    return line, cost[-1]


def plot_for_k(k):
    bigtable = BigtablePolicy(flush_size, k)
    binomial = BinomialPolicy(flush_size, k)
    constant = ConstantPolicy(flush_size, k)
    if k <= 6:
        exploring = ExploringPolicy(flush_size, k, c=2, d=10)
    else:
        exploring = ExploringPolicy(flush_size, k, c=3, d=10)
    minlatency = MinLatencyPolicy(flush_size, k)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    l1, m1 = simulate(ax, max_flushes, bigtable, "#70AD47")
    l2, m2 = simulate(ax, max_flushes, binomial, "#FF0000")
    l3, m3 = simulate(ax, max_flushes, constant, "#ED7D31")
    l4, m4 = simulate(ax, max_flushes, exploring, "#7030A0")
    l5, m5 = simulate(ax, max_flushes, minlatency, "#5B9BD5")
    ax.set_xlim(0, max_flushes)
    if k ==3:
        ax.set_ylim(0, max([m2, m5]) * 1.05)
    else:
        ax.set_ylim(0, max([m2, m5]) * 1.5)
    ax.set_xlabel("Total Flushes (Thousands)")
    ax.set_ylabel("Total Write Cost (GB)")
    ax.legend([l1, l2, l3, l4, l5],
              [policies["google-default"], policies["binomial"], policies["constant"],
               policies["exploring"], policies["min-latency"]],
              bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=5, mode="expand", borderaxespad=0.)
    ax.grid()
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))
    plt.tight_layout()
    plt.savefig("sim_cost_" + str(k) + ".png", transparent=True, bbox_inches="tight")
    print("Done k=" + str(k))


if __name__ == '__main__':
    for k in range(3, 11):
        plot_for_k(k)
