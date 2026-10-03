# Rules-based ETF strategy research

A Python research project that tests whether low-turnover, rules-based ETF
strategies can match or beat buying and holding the S&P 500 (SPY) **on a
risk-adjusted basis, after costs and taxes**, with a focus on limiting losses
in downturns.

**Scope: backtesting and paper trading only. This project never places
real-money trades.**

> Most published market-beating strategies fail out of sample. This project
> is built to *measure*, not to sell a strategy. Buying and holding an S&P 500
> index fund is a perfectly valid conclusion.

## Status

| Step | Contents | Status |
|---|---|---|
| 1 | Data layer: downloaders, cache, validation | **done** |
| 2 | Backtest engine, cost model, tax-lot model, metrics | **done** |
| 3 | Strategies 0-6 | not started |
| 4 | Walk-forward, Deflated Sharpe, PBO, bootstrap | not started |
| 5 | Charts and report | not started |
| 6 | Alpaca paper trading (paper endpoint only) | not started |

## Approved research decisions (2026-10-03)

| Decision | Choice |
|---|---|
| Goal | Similar return to SPY with smaller drawdowns (rank by risk-adjusted metrics) |
| Accounts | Report taxable, Roth IRA and traditional IRA |
| Tax rates (illustrative) | Federal: short-term/ordinary 24%, long-term 15%, qualified dividends 15%. State: 5% flat; Treasury interest state-exempt |
| Taxable-account rules | FIFO lots; taxes paid yearly from the portfolio; losses carried forward; no $3k ordinary offset; report "still holding" and "liquidated at end" |
| Starting capital | $100,000 |
| Data | Tiingo free tier (primary), yfinance (cross-check only), FRED DTB3 (cash), Kenneth French library (long-history checks) |
| Paper broker | Alpaca paper trading via `alpaca-py` |
| Benchmark | SPY, dividend-adjusted total return |
| Held-out period | 2021-10-01 onward, locked until the final configuration is approved |
| Fund choices | EFA (not VEU), AGG (not BND), BIL (not SHV); FRED T-bill rate before BIL launched |
| Proxies | None. ETFs only, from their own inception dates |
| Execution | Signal at the month-end close, fill at the next trading day's close |

All of these live in [`config/settings.yaml`](config/settings.yaml).

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"        # core + yfinance cross-check + alpaca-py + pytest
cp .env.example .env           # then fill in TIINGO_API_KEY
```

Get a free Tiingo API token at <https://www.tiingo.com> (Account > API).
FRED and the Kenneth French library need no key.

Secrets are read only from environment variables (or a local `.env`, which is
git-ignored). They are never printed: log output passes through a filter that
masks any secret value, and the Tiingo key is sent in a request header rather
than the URL.

## How to run

```bash
# Download everything (Tiingo prices, FRED T-bill rates, French data, yfinance cross-check)
python -m src.data download

# Only some symbols, ignoring the cache
python -m src.data download --symbols SPY XLK --refresh

# Data-quality checks; writes logs/data_validation_<date>.md and .csv
python -m src.data validate

# Tests (all network calls are mocked; real network access is blocked)
pytest
```

The full enabled universe is 23 symbols, which is well within Tiingo's free tier
(about 500 unique symbols per month, 50 requests per hour, 1,000 per day).
Request counts are stored in `data/cache/_state/` so separate runs share one budget.
Each symbol needs one request per download.

## Project layout

```
config/settings.yaml   universes, dates, costs, tax rates, strategy parameters
src/config.py          settings + secret loading, redaction
src/logging_setup.py   logging with secret masking
src/data/              Tiingo, yfinance, FRED, French downloaders; Parquet cache;
                       validation; DataStore (inception masking, held-out lock)
src/strategies/        (step 3) one module per strategy -> target weights
src/backtest/          engine, cost model, tax lots, metrics, MarketData builder
src/validation/        (step 4) walk-forward, Deflated Sharpe, PBO, bootstrap
src/reporting/         (step 5) charts and report
src/paper/             (step 6) Alpaca paper client and rebalance job
tests/                 pytest suite
docs/heldout_log.md    written log of every held-out evaluation
notebooks/             optional; no untested logic allowed here
```

## How the data layer protects the results

* **No survivorship bias from stock lists.** Only ETFs and indexes are used,
  never today's S&P 500 members.
* **No data before inception.** Each ETF's inception date is recorded in the
  config. Rows dated before it are dropped, and nothing is back-filled. Before
  BIL existed (May 2007), cash earns the FRED 3-month T-bill rate, and every
  day's cash return is labeled with its source.
* **Limited filling.** Interior gaps of up to 3 trading days are
  forward-filled. Leading and trailing gaps never are.
* **Held-out lock.** Analysis data on or after 2021-10-01 is withheld. Asking
  for it raises `HeldOutLockedError` until `dates.heldout_unlocked` is set,
  which happens only after you approve the final configuration and record the
  date in [`docs/heldout_log.md`](docs/heldout_log.md).
* **Validation.** The checks cover missing trading days (measured against
  SPY's calendar), long gaps, zero, negative or missing prices, daily moves
  above 20%, runs of unchanged closes, data that has stopped updating,
  first-date versus inception mismatches, and Tiingo versus yfinance
  disagreements in daily and monthly adjusted returns. Checks only flag
  problems for review. They never change data.

## How the backtest engine works

* **Prices.** The engine uses total-return (dividend- and split-adjusted)
  closes, so dividends are reinvested in the same fund on the ex-date at no
  cost. The benchmark is SPY bought once and held.
* **Timing.** A target set on a signal date is filled at the close one
  trading day later (`execution.execution_lag_days`). The engine reads only
  prices up to the current day; a test changes later prices and confirms
  earlier values don't move.
* **Trades.** Each execution trades every position to its target weight of
  the portfolio value net of that execution's costs. Trades under $1 are
  skipped. Fractional units are allowed by default; a whole-share mode
  leaves the remainder as cash.
* **Costs.** Every trade pays 5 bp (basis points) per side for spread plus
  slippage, with sensitivity runs at 2 and 10 bp. There is no commission.
  Sales also pay the SEC Section 31 fee ($20.60 per $1M) and the FINRA
  trading activity fee ($0.000195 per share, capped at $9.79 per trade).
  Cost drag is reported as costs per year divided by average portfolio value.
* **Cash.** BIL before May 2007 is a synthetic $1-price fund paying the FRED
  T-bill rate as daily interest. Every report labels it. Trading costs apply
  to it like any ETF, which is slightly conservative.
* **Taxable account.** Positions are tracked in FIFO lots:
  * A gain is long-term only if the fund was held more than one year.
  * Each reinvested dividend becomes its own lot.
  * Qualified-fund dividends must pass the IRS holding test: held more than
    60 days in the 121-day window starting 60 days before the ex-date.
    Dividends that fail it are taxed as ordinary income.
  * Each year's short- and long-term results are netted as on Schedule D,
    and net losses carry forward.
  * Each year's tax is paid on the first trading day on or after the next
    April 15 by selling holdings pro rata. Those sales can realize further
    gains.
  * At the end, after-tax value is reported two ways. "Still holding"
    subtracts taxes owed but unpaid. "Liquidated" also sells everything,
    paying costs and tax on those gains.
* **IRAs.** The Roth IRA result is the pre-tax result. The traditional IRA
  result applies a 29% withdrawal tax to the ending value.
* **Wash sales** are flagged whenever a losing sale has a purchase of the
  same or a substantially identical fund within 30 days before or after it.
  Flags come in two kinds: triggered by a strategy trade, or by dividend
  reinvestment. Disallowed losses are *not* adjusted, so taxable results for
  flagged strategies are slightly optimistic.

## Limitations of free data and of this study

* **Adjusted prices are vendor-specific.** Tiingo and Yahoo adjust for
  dividends slightly differently, so daily returns disagree a little around
  ex-dividend dates. Monthly differences above 0.2% are flagged. Tiingo
  re-bases adjusted history whenever a dividend is paid, so the cache always
  re-downloads full history instead of appending.
* **Short histories.** Strategies are tested only from the point where every
  ETF they need exists and the lookback period is filled:
  * Sector strategies start around late 1999. The sector ETFs launched in
    December 1998, so the start just covers the 2000–2002 bear market.
  * GEM starts around late 2004, because EFA and AGG need 12 months of history.
  * GTAA starts around early 2007, because of DBC plus a 10-month moving average.
  * The factor blend starts around 2014.

  **GEM, GTAA and the factor blend therefore miss the 2000–2002 bear market
  entirely, and the factor blend also misses 2008–2009.**
* **The Kenneth French industry portfolios are gross and non-investable.**
  They carry no fees, costs or taxes, and could not have been bought as funds
  for most of their history. They are labeled that way everywhere and serve
  only as a long-history sanity check.
* **T-bill conversion.** FRED's DTB3 is a discount-basis yield. It is
  converted to a daily return using the 91-day bill price, applying the rate
  known at the start of each period. Real T-bill ETF returns are slightly
  lower because of fees.
* **Tax simplifications** (applied from step 2 onward):
  * All equity-ETF dividends are treated as qualified.
  * REIT (VNQ) distributions are treated as ordinary income, ignoring the
    Section 199A deduction.
  * Bond and T-bill fund distributions are treated as ordinary income.
  * DBC is a commodity pool that issues a K-1 with 60/40 mark-to-market
    treatment. That is **not** modeled: DBC is taxed like an ordinary ETF.
  * Tax rates are flat and illustrative, not your actual brackets.
  * Each reinvested dividend in the same fund and month is merged into one
    lot dated at the later date. This can only make a gain short-term,
    never long-term, so it is the conservative direction.
  * The dividend holding test treats a lot that is still held as held for
    the whole window. A lot partly sold is split exactly by the fraction sold.
  * Unused loss carryforwards at the end of the backtest are given no value.
  * The SEC and FINRA fees use 2026 rates for the whole history. Historical
    rates differed, but the effect is well under 1 bp per year.
* **yfinance** is an unofficial wrapper intended for personal use. It is used
  only as a cross-check, with caching and a 2-second delay between requests.
* **Few independent trials.** With 6 candidate strategies, the Probability of
  Backtest Overfitting estimate will be noisy.
* **Past performance does not predict future results.** A backtest is a
  measurement under assumptions, not a forecast.

## Safety

* No live-trading code exists. The paper client (step 6) will refuse to start
  unless `PAPER_TRADING=true` and the base URL is Alpaca's paper endpoint.
* A future live module would need its own explicitly named environment flag
  plus an interactive confirmation, and would be disabled by default.
* This project never connects to Robinhood in any form.
