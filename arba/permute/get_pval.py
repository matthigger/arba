from bisect import bisect_right


def get_pval(stat, stat_null, sort_flag=True, stat_include_flag=True):
    """ comptues pval of stat given stats from the null

    Args:
        stat (float): statistic to test
        stat_null (list): stats drawn from the null hypothesis (e.g. perm test)
        sort_flag (bool): if false, indicated stat null is already sorted
        stat_include_flag (bool): toggles whether to add a copy of stat into
                                  stat_null (Edgington 1969)
    Returns:
          pval (float): percentage of stat_null which is smaller than stat
    """

    num_perm = len(stat_null)

    if sort_flag:
        num_leq = sum(s <= stat for s in stat_null)
    else:
        num_leq = bisect_right(stat_null, stat)

    if stat_include_flag:
        num_perm += 1
    else:
        num_leq -= 1
        assert stat in stat_null, 'stat not found in stat_null'

    return 1 - num_leq / num_perm
