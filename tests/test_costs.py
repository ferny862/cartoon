import pytest

from src.backtest.costs import CostModel


def test_buy_pays_spread_only():
    m = CostModel(spread_slippage_bps=5, sec_fee_per_million=20.6, finra_taf_per_share=0.000195, finra_taf_max_per_trade=9.79)
    c = m.trade_cost(10_000, shares=25)
    assert c.spread == pytest.approx(5.0)          # 10,000 * 0.0005
    assert c.sec_fee == 0 and c.finra_taf == 0 and c.total == pytest.approx(5.0)


def test_sale_pays_regulatory_fees():
    m = CostModel(spread_slippage_bps=5, sec_fee_per_million=20.6, finra_taf_per_share=0.000195, finra_taf_max_per_trade=9.79)
    c = m.trade_cost(-50_000, shares=100)
    assert c.spread == pytest.approx(25.0)
    assert c.sec_fee == pytest.approx(50_000 * 20.6 / 1e6)  # $1.03
    assert c.finra_taf == pytest.approx(0.0195)
    assert c.total == pytest.approx(25.0 + 1.03 + 0.0195)


def test_taf_cap():
    m = CostModel(spread_slippage_bps=0, finra_taf_per_share=0.000195, finra_taf_max_per_trade=9.79)
    assert m.trade_cost(-1e7, shares=1e6).finra_taf == pytest.approx(9.79)


def test_commission_and_zero_trade():
    m = CostModel(spread_slippage_bps=2, commission_per_trade=1.0)
    assert m.trade_cost(0).total == 0
    assert m.trade_cost(1000).total == pytest.approx(0.2 + 1.0)


def test_from_settings_uses_verified_fees(settings):
    m = CostModel.from_settings(settings)
    assert m.spread_slippage_bps == 5
    assert m.sec_fee_per_million == pytest.approx(20.60)
    assert m.finra_taf_per_share == pytest.approx(0.000195)
    assert m.finra_taf_max_per_trade == pytest.approx(9.79)
    assert m.with_spread(10).spread_slippage_bps == 10
    assert CostModel.from_settings(settings, spread_slippage_bps=2).spread_slippage_bps == 2
