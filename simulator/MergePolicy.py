#
# This file contains classes for generating flush size and merge policies.
#    FlushGenerator: Base class for generating flush size.
#      ConstantFlush: Class for generating constant flush size.
#      RandomFlush: Class for generating random flush size.
#
#    MergePolicy: Base class for merge policy.
#      BigtablePolicy: Bigtable (Google default) merge policy.
#      BinomialPolicy: Binomial merge policy.
#      ConstantPolicy: Constant merge policy (in AsterixDB prioir to 0.9.4)
#      ExploringPolicy: Exploring (default in HBase) merge policy.
#      MinLatencyPolicy: MinLatency merge policy.
#      PrefixPolicy: Prefix (default in AsterixDB) merge policy.
#


import math
import random
from functools import lru_cache
from itertools import accumulate


class FlushGenerator(object):
    """ Base class to generate a flush size """
    def __init__(self):
        pass

    def next(self):
        """ Return the next size for flush """
        return 0

    def reset(self):
        """ Reset """
        pass


class ConstantFlush(FlushGenerator):
    """ Generator for constant size (c) flush  """
    def __init__(self, c):
        if type(c) == int:
            if c > 0:
                self.c = c
            else:
                raise ValueError("Input must be a positive integer")
        else:
            raise TypeError("Input must be a valid integer")

    def next(self):
        return self.c


class RandomFlush(FlushGenerator):
    """ Generate flush size randomly between rmin and rmax """
    def __init__(self, rmin, rmax):
        if type(rmin) != int or type(rmax) != int:
            raise TypeError("Inputs must be valid integers")
        if rmin <= 0:
            raise ValueError("rmin must be a positive integer")
        if rmax <= 0:
            raise ValueError("rmax must be a positive integer")
        if rmax < rmin:
            raise ValueError("rmax must not be smaller than min")
        self.rmin = rmin
        self.rmax = rmax

    def range(self):
        return self.rmin, self.rmax

    def next(self):
        return random.randint(self.rmin, self.rmax)


class MergePolicy(object):
    """ Base class for merge policy.

    Attributes:
        fs: Flush size. fs must be either a positive integer (fixed size flush), or a FlushGenerator object.
        ratio: Size of a merge's outputs over its inputs.
    """

    def __init__(self, fs, ratio=1.0):
        if type(fs) == int:
            if fs < 1:
                raise ValueError("fs must be a positive integer or FlushGenerator type")
            else:
                self.generator = ConstantFlush(fs)
        elif isinstance(fs, FlushGenerator):
            self.generator = fs
        else:
            raise TypeError("fs must be a positive integer or FlushGenerator type")
        if type(ratio) != int and type(ratio) != float:
            raise TypeError("ratio must be a positive number")
        if math.isnan(ratio) or math.isinf(ratio) or ratio <= 0.0:
            raise ValueError("ratio must be a positive number")
        self.comps = []
        self.ratio = ratio
        self.fcnt = 0
        self.mcnt = 0

    @staticmethod
    def policy_name():
        """ Name of the policy """
        return ""

    def reset(self):
        """ Reset to empty """
        self.comps = []
        self.fcnt = 0
        self.mcnt = 0
        self.generator.reset()

    def components(self):
        """ List of component sizes, in the order of newer to older """
        return tuple(self.comps)

    def flush_count(self):
        """ The number of flushes so far """
        return self.fcnt

    def merge_count(self):
        """ The number of merges so far """
        return self.mcnt

    def flush(self):
        """ Perform a flush operation

        Return:
            (flush_size, start_index, in_comps, out_comps)
              - flush_size: Size of the flushed component
              - start_index: The index of the first component to be merged, -1 if no merge
              - in_comps: Tuple of components to be merged, () if no merge
              - out_comps: Tuple of componenets created from the merge, () if no merge
        """
        flush_size = self.generator.next()
        self.comps.insert(0, flush_size)
        self.fcnt += 1
        start_idx, in_comps, out_comps = self.merge()
        return flush_size, start_idx, in_comps, out_comps

    def merge(self):
        """ Perform a merge operation

        Return:
            (start_index, in_comps, out_comps)
              - start_index: The index of the first component to be merged, -1 if no merge
              - in_comps: Tuple of components to be merged, () if no merge
              - out_comps: Tuple of componenets created from the merge, () if no merge
        """
        return -1, (), ()


class BigtablePolicy(MergePolicy):
    """ Bigtable (Google default) merge policy.

    Attributes:
        fs: Flush size. fs must be either a positive integer (fixed size flush), or a FlushGenerator object.
        k: Maximum number of components.
        ratio: Size of a merge's outputs over its inputs.
    """

    def __init__(self, fs, k, ratio=1.0):
        if type(k) != int:
            raise TypeError("k must be a valid integer")
        if k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k
        super(BigtablePolicy, self).__init__(fs, ratio)

    @staticmethod
    def policy_name():
        """ Name of the policy """
        return "Bigtable"

    def merge(self):
        """ Perform a merge operation

        Return:
            (start_index, in_comps, out_comps)
              - start_index: The index of the first component to be merged, -1 if no merge
              - in_comps: Tuple of components to be merged, () if no merge
              - out_comps: Tuple of componenets created from the merge, () if no merge
        """
        cnt = len(self.comps)
        # No merge
        if cnt <= self.k:
            return -1, (), ()
        comps = list(reversed(self.comps))
        newer = list(reversed(list(accumulate(self.comps))))
        for merge_idx in range(cnt-1):
            if comps[merge_idx] <= newer[merge_idx + 1]:
                break
        del newer
        merge_idx = cnt - merge_idx - 1
        mergable_comps = tuple(self.comps[0:merge_idx+1])
        new_comp = int(math.ceil(sum(mergable_comps) * self.ratio))
        self.comps = [new_comp,] + self.comps[merge_idx+1:]
        self.mcnt += 1
        return 0, mergable_comps, (new_comp,)


class BinomialPolicy(MergePolicy):
    """ Binomial merge policy.

    Attributes:
        fs: Flush size. fs must be either a positive integer (fixed size flush), or a FlushGenerator object.
        k: Maximum number of components.
        ratio: Size of a merge's outputs over its inputs.
    """

    def __init__(self, fs, k):
        if type(k) != int:
            raise TypeError("k must be a valid integer")
        if k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k
        super(BinomialPolicy, self).__init__(fs)

    @staticmethod
    def policy_name():
        """ Name of the policy """
        return "Binomial"

    def merge(self):
        cnt = len(self.comps)
        if cnt <= self.k:
            return -1, (), ()

        def binomial_choose(n, k):
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1
            w = n + 1
            bin = [0, ] * (w ** 2)

            def cell(row, col):
                return row * w + col

            for r in range(0, w):
                for c in range(0, min(r, k)+1):
                    if c == 0 or c == r:
                        bin[cell(r,c)] = 1
                    else:
                        bin[cell(r,c)] = bin[cell(r-1,c-1)] + bin[cell(r-1,c)]
            return bin[cell(n,k)]

        def binomial_index(d, h, t):
            if t == 0:
                return 0
            if t < binomial_choose(d +h-1, h):
                return binomial_index(d-1, h, t)
            return binomial_index(d, h-1, t - binomial_choose(d+h-1, h)) + 1

        def tree(d):
            if d < 0:
                return 0
            return tree(d-1) + binomial_choose(d + min(d, self.k) - 1, d)

        depth = 0
        while tree(depth) < self.fcnt:
            depth += 1
        merge_idx = binomial_index(depth, min(depth, self.k) - 1, self.fcnt - tree(depth-1) - 1)
        if merge_idx == cnt - 1:
            return -1, (), ()
        merge_idx = cnt - merge_idx - 1  # Might have some problem here
        mergable_comps = tuple(self.comps[0:merge_idx+1])
        new_comp = int(math.ceil(sum(mergable_comps) * self.ratio))
        self.comps = [new_comp,] + self.comps[merge_idx+1:]
        self.mcnt += 1
        return 0, mergable_comps, (new_comp,)


class ConstantPolicy(MergePolicy):
    """ Constant merge policy.

    Attributes:
        fs: Flush size. fs must be either a positive integer (fixed size flush), or a FlushGenerator object.
        k: Maximum number of components.
        ratio: Size of a merge's outputs over its inputs.
    """

    def __init__(self, fs, k, ratio=1.0):
        if type(k) != int:
            raise TypeError("k must be a valid integer")
        if k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k
        super(ConstantPolicy, self).__init__(fs, ratio)

    @staticmethod
    def policy_name():
        """ Name of the policy """
        return "Constant"

    def merge(self):
        """ Perform a merge operation

        Return:
            (start_index, in_comps, out_comps)
              - start_index: The index of the first component to be merged, -1 if no merge
              - in_comps: Tuple of components to be merged, () if no merge
              - out_comps: Tuple of componenets created from the merge, () if no merge
        """
        if len(self.comps) > self.k:
            mergable_comps = tuple(self.comps)
            new_comp = int(math.ceil(sum(mergable_comps) * self.ratio))
            self.comps = [new_comp,]
            self.mcnt += 1
            return 0, mergable_comps, (new_comp,)
        else:
            return -1, (), ()


class ExploringPolicy(MergePolicy):
    """ Exploring merge policy.

    Attributes:
        fs: Flush size. fs must be either a positive integer (fixed size flush), or a FlushGenerator object.
        c: ?
        d: ?
        lambda_: ?
        ratio: Size of a merge's outputs over its inputs.
    """

    def __init__(self, fs, k, c=3, d=10, lambda_=1.2, ratio=1.0):
        if type(k) != int:
            raise TypeError("k must be a valid integer")
        if k < 1:
            raise ValueError("k must be a positive integer")
        if type(c) != int:
            raise TypeError("c must be a valid integer")
        if c < 1:
            raise ValueError("c must be a positive integer")
        if type(d) != int:
            raise TypeError("d must be a valid integer")
        if d < 1:
            raise ValueError("d must be a positive integer")
        if type(lambda_) != int and type(lambda_) != float:
            raise TypeError("lambda_ must be a positive number")
        if lambda_ <= 0.0:
            raise ValueError("lambda_ must be a positive number")
        self.k = k
        self.c = c
        self.d = min(d, k)
        self.lambda_ = lambda_
        super(ExploringPolicy, self).__init__(fs, ratio)

    @staticmethod
    def policy_name():
        """ Name of the policy """
        return "Exploring"

    def merge(self):
        """ Perform a merge operation

        Return:
            (start_index, in_comps, out_comps)
              - start_index: The index of the first component to be merged, -1 if no merge
              - in_comps: Tuple of components to be merged, () if no merge
              - out_comps: Tuple of componenets created from the merge, () if no merge
        """
        memoize = lru_cache(None)

        comps = list(reversed(self.comps))
        cnt = len(comps)
        might_be_stuck = cnt > self.k

        @memoize
        def min_size(i, j):
            return math.inf if j <= i else min(min_size(i, j-1), comps[j-1])

        @memoize
        def max_size(i, j):
            return -math.inf if j <= i else max(max_size(i, j-1), comps[j-1])

        @memoize
        def cum_size(i, j):
            return 0 if j <= i else cum_size(i, j-1) + comps[j-1]

        def ave_size(i, j):
            return cum_size(i, j) / (j-i)

        def candidates():
            for i in range(cnt):
                for j in range(i + self.c, min(i + self.d, cnt+1)):
                    if max_size(i, j) <= self.lambda_ * (cum_size(i, j) - max_size(i, j)):
                        yield i, j

        if might_be_stuck:
            best = min(candidates(), key=lambda ij: ave_size(*ij), default=None)
        else:
            best = max(
                candidates(),
                key=lambda ij: (ij[1] - ij[0], -cum_size(*ij)),
                default=None,
            )

        if best is not None:
            i, j = best
        elif might_be_stuck:
            def candidates():
                for i in range(cnt - self.c + 1):
                    yield i, i+self.c
            i, j = min(candidates(), key=lambda ij: cum_size(*ij))
        else:
            i, j = cnt-1, cnt

        if j-i < 2:
            return -1, (), ()  # No need to merge
        new_comp = int(math.ceil(cum_size(i, j) * self.ratio))
        i = cnt - i
        j = cnt - j
        mergable_comps = tuple(self.comps[j:i])
        self.comps = self.comps[:j] + [new_comp,] + self.comps[i:]
        self.mcnt += 1
        return j, mergable_comps, (new_comp,)


class MinLatencyPolicy(MergePolicy):
    """ MinLatency merge policy.

    Attributes:
        fs: Flush size. fs must be either a positive integer (fixed size flush), or a FlushGenerator object.
        k: Maximum number of components.
        ratio: Size of a merge's outputs over its inputs.
    """

    def __init__(self, fs, k, ratio=1.0):
        if type(k) != int:
            raise TypeError("k must be a valid integer")
        if k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k
        super(MinLatencyPolicy, self).__init__(fs, ratio)

    @staticmethod
    def policy_name():
        """ Name of the policy """
        return "MinLatency"

    def merge(self):
        cnt = len(self.comps)
        if cnt <= self.k:
            return -1, (), ()

        def binomial_choose(n, k):
            if k < 0 or k > n:
                return 0
            if k == 0 or k == n:
                return 1
            w = n + 1
            bin = [0, ] * (w ** 2)

            def cell(row, col):
                return row * w + col

            for r in range(0, w):
                for c in range(0, min(r, k)+1):
                    if c == 0 or c == r:
                        bin[cell(r,c)] = 1
                    else:
                        bin[cell(r,c)] = bin[cell(r-1,c-1)] + bin[cell(r-1,c)]
            return bin[cell(n,k)]

        def binomial_index(d, h, t):
            if t == 0:
                return 0
            if t < binomial_choose(d +h-1, h):
                return binomial_index(d-1, h, t)
            return binomial_index(d, h-1, t - binomial_choose(d+h-1, h)) + 1

        def tree(d):
            if d < 0:
                return 0
            return binomial_choose(d + self.k, d) - 1

        depth = 0
        while tree(depth) < self.fcnt:
            depth += 1
        merge_idx = binomial_index(depth, self.k - 1, self.fcnt - tree(depth-1) - 1)
        if merge_idx == cnt - 1:
            return -1, (), ()
        merge_idx = cnt - merge_idx - 1
        mergable_comps = tuple(self.comps[0:merge_idx+1])
        new_comp = int(math.ceil(sum(mergable_comps) * self.ratio))
        self.comps = [new_comp,] + self.comps[merge_idx+1:]
        self.mcnt += 1
        return 0, mergable_comps, (new_comp,)


class NoMergePolicy(MergePolicy):
    """ No merge policy """
    def __init__(self, fs):
        super(NoMergePolicy, self).__init__(fs, 1.0)

    @staticmethod
    def policy_name():
        """ Name of the policy """
        return "NoMerge"


class PrefixPolicy(MergePolicy):
    """ Prefix merge policy.

    Attributes:
        fs: Flush size. fs must be either a positive integer (fixed size flush), or a FlushGenerator object.
        k: Maximum number of components.
        m: max-mergable-component-size
        c: max-tolerance-component-count
        r: max-mergable-component-size-ratio
        ratio: Size of a merge's outputs over its inputs.
    """

    def __init__(self, fs, m=10**6, c=5, r=1.2, ratio=1.0):
        if type(m) != int:
            raise TypeError("m must be a valid integer")
        if m < 1:
            raise ValueError("m must be a positive integer")
        if type(c) != int:
            raise TypeError("c must be a valid integer")
        if c < 1:
            raise ValueError("c must be a positive integer")
        if type(r) != int and type(r) != float:
            raise TypeError("r must be a valid number")
        if r <= 0.0:
            raise ValueError("r must be a positive number")
        self.m = m
        self.c = c
        self.r = r
        super(PrefixPolicy, self).__init__(fs, ratio)

    @staticmethod
    def policy_name(self):
        """ Name of the policy """
        return "Prefix"

    def merge(self):
        """ Perform a merge operation

        Return:
            (start_index, in_comps, out_comps)
              - start_index: The index of the first component to be merged, -1 if no merge
              - in_comps: Tuple of components to be merged, () if no merge
              - out_comps: Tuple of componenets created from the merge, () if no merge
        """
        memoize = lru_cache(None)

        comps = list(reversed(self.comps))
        cnt = len(comps)

        @memoize
        def cum_size(i, j):
            return 0 if j <= i else cum_size(i, j-1) + comps[j-1]

        @memoize
        def max_size(i, j):
            return -math.inf if j <= i else max(max_size(i, j-1), comps[j-1])

        def find_merge():
            for i in range(cnt):
                for j in range(i+1, cnt+1):
                    if max_size(i, j) > self.m:
                        break
                    if j-i > self.c or cum_size(i, j) > self.m:
                        if comps[i] < self.r * cum_size(i+1, j):
                            return i, j
                        break
            return cnt-1, cnt

        i, j = find_merge()
        if j-1 < 2:
            return -1, (), ()  # No need to merge
        new_comp = int(math.ceil(cum_size(i, j) * self.ratio))
        i = cnt - i
        j = cnt - j
        mergable_comps = tuple(self.comps[j:i])
        self.comps = self.comps[:j] + [new_comp,] + self.comps[i:]
        self.mcnt += 1
        return j, mergable_comps, (new_comp,)
