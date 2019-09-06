from math import isclose

from arba.permute import get_pval


def test_get_pval0(stat=10,
                   stat_null=range(10),
                   sort_flag=True,
                   stat_include_flag=True):
    pval_out = get_pval(stat=stat,
                        stat_null=stat_null,
                        sort_flag=sort_flag,
                        stat_include_flag=stat_include_flag)
    pval_expected = 1 / 11
    assert isclose(pval_out, pval_expected), 'test case 0 fail'


def test_get_pval1(stat=10,
                   stat_null=range(10),
                   sort_flag=False,
                   stat_include_flag=True):
    pval_out = get_pval(stat=stat,
                        stat_null=stat_null,
                        sort_flag=sort_flag,
                        stat_include_flag=stat_include_flag)
    pval_expected = 1 / 11
    assert isclose(pval_out, pval_expected), 'test case 0 fail'


def test_get_pval2(stat=10,
                   stat_null=range(11),
                   sort_flag=False,
                   stat_include_flag=False):
    pval_out = get_pval(stat=stat,
                        stat_null=stat_null,
                        sort_flag=sort_flag,
                        stat_include_flag=stat_include_flag)
    pval_expected = 1 / 11
    assert isclose(pval_out, pval_expected), 'test case 0 fail'
