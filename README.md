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
| 3 | Strategies 0-6, runner, trial log, French sanity check | **done** |
| 4 | Walk-forward, Deflated Sharpe, PBO, bootstrap | **done** |
| 5 | Charts, downturn analysis, pre-registered verdict, HTML/Markdown report | **done** |
| 6 | Alpaca paper trading (paper endpoint only, dry run by default) | **done** |

## Results so far

* **In-sample (1993 to Sep 2021):** no strategy met the pre-registered bar.
  The trend filter and sector momentum with a market filter roughly matched
  SPY's return with much smaller crash losses, but the difference could not
  be told apart from luck.
* **Held-out test (Oct 2021 to Oct 2026, run once on 2026-10-05, see
  `docs/heldout_log.md`):** both of those strategies returned about 6–7% a
  year against SPY's 13.6%. Neither avoided the 2022 bear market; both were
  whipsawed in it. The trend filter's shortfall is statistically clear.
  **Conclusion: buy and hold an S&P 500 index fund.** The held-out set is
  now spent; any later change is post-held-out and must be labeled that way.

Reproduce with `python -m src.reporting` (in-sample, all strategies) and
`python -m src.validation.heldout --strategies trend_faber sector_mom_market_filter`.

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

**In a Claude Code cloud environment**, store the token as an environment API
credential instead of in `.env`:
* Credential type: **Bearer**
* Allowed website: `api.tiingo.com`
* Header name `Authorization`, with the prefix changed from `Bearer` to
  **`Token`**, and your Tiingo key as the value. The resolved request should
  carry `Authorization: Token <key>`, which is the form Tiingo expects.

The proxy then adds the key to every Tiingo request, and the key never
reaches the session. With no `TIINGO_API_KEY` set, the client sends no key of its own and relies on
the injected header automatically (`data.tiingo.credential: auto`).
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

# Backtests (in-sample only until the held-out period is unlocked)
python -m src.backtest run                                   # all strategies, 5 bp, with taxes
python -m src.backtest run --strategies gem --cost-bps 2 5 10
python -m src.backtest french                                # strategies 5-6 on French industries

# Validation: walk-forward windows, regimes, Deflated Sharpe, PBO, bootstrap
python -m src.validation                                     # writes logs/validation_<date>/

# Full report: reports/output/report_<date>.html and summary_<date>.md
python -m src.reporting

# Tests (all network calls are mocked; real network access is blocked)
pytest
```

Every backtest run is appended to `logs/trials.jsonl`. A new set of strategy
parameters counts as a new trial for the overfitting statistics. Re-running
the same parameters with different costs or account types does not.

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
src/strategies/        one module per strategy -> target weights per signal date
src/backtest/          engine, cost model, tax lots, metrics, MarketData builder
src/validation/        (step 4) walk-forward, Deflated Sharpe, PBO, bootstrap
src/reporting/         charts, downturns, pre-registered verdict, HTML/Markdown report
src/paper/             Alpaca paper client, safety guards, monthly rebalance job
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

## Strategies as implemented

All signals are computed at the last trading day's close of each month,
using total-return (dividend-adjusted) closes, and filled at the next
trading day's close. Each strategy only ever sees month-end closes up to the
signal date; the base class enforces this by truncating the history. A test
changes future prices for every strategy and checks that past signals don't
move. "Above the SMA" means strictly above the simple moving average of the
last 10 month-end closes, including the current one. Cash is BIL, or the
FRED T-bill series before BIL existed.

| # | Name | Rule |
|---|---|---|
| 0 | `benchmark` | Buy SPY once and hold it, dividends reinvested. |
| 1 | `trend_faber` | 100% SPY if SPY is above its 10-month SMA, else 100% cash. |
| 2 | `gem` | If SPY's 12-month return beats T-bills' 12-month return, hold the stronger of SPY and EFA over 12 months; otherwise hold AGG. |
| 3 | `gtaa5` | 20% each in SPY, EFA, IEF, VNQ, DBC; a sleeve goes to cash when its fund is not above its 10-month SMA. |
| 4 | `factor_blend` | 25% each in QUAL, MTUM, VLUE, USMV. Rebalance at each December month-end, or at any month-end when a weight has drifted more than 5 percentage points. |
| 5 | `sector_mom_sector_filter` | Rank the 9 original sector SPDRs by 12-1 momentum and hold the top 3 at one-third each. A held sector that is not above its own 10-month SMA has its third moved to cash. |
| 6 | `sector_mom_market_filter` | Same ranking and top 3, but everything moves to cash when SPY is not above its 10-month SMA. |

* **12-1 momentum** here is the return from 14 month-ends ago to last
  month-end, `P[t-1] / P[t-13] - 1`: 12 monthly returns, skipping the most
  recent month. Some academic papers use 11 returns (`P[t-1] / P[t-12]`);
  that would be a parameter change requiring approval.
* **Ties in the sector ranking** are broken by the order sectors are listed
  in the config.
* **Factor-blend drift** is measured from month-end closes since the last
  rebalance signal. Actual fills happen one day later, so the engine's
  drift can differ slightly.
* **Long-history sanity check.** Strategies 5 and 6 are also run with the
  same rules on Kenneth French's daily 10-industry portfolios from 1926, using
  French's risk-free rate as cash and French's market return for the market
  filter. These runs have no costs or taxes, and the results are **gross and
  non-investable**. They choose the top 3 of 10 industries, not 9 sectors.

## How the validation works

All checks use **monthly** returns, in-sample only (before 2021-10-01). The
first, partial month of each strategy is dropped.

* **Walk-forward / sub-periods.** No strategy fits parameters to past data,
  so walk-forward analysis here means checking whether results against SPY
  hold up in separate periods. There are three tables:
  * Fixed non-overlapping windows: 2000–04, 2005–09, 2010–14, 2015–19 and
    2020 to September 2021.
  * Expanding windows that start at each strategy's own start and grow 5
    years at a time.
  * The named regimes.

  A strategy that starts partway through a window is compared from its own
  start against SPY bought the same day. Those rows are marked `partial`,
  with the actual dates. Windows covering less than about 10 months (3 for
  the regimes) are marked `not live`. The 2022 regime shows
  `held out (locked)` until the held-out period is unlocked.
* **Deflated Sharpe Ratio** (Bailey and López de Prado, 2014). This is the
  probability that a strategy's true Sharpe ratio beats the best Sharpe
  ratio that luck alone would produce, given how many configurations were
  tried. It also accounts for skew, fat tails and the length of the record.
  The number of configurations comes from `logs/trials.jsonl` (ETF universe
  only). The spread of Sharpe ratios across configurations is measured
  across the current candidates, each over its own history. 0.95 or above
  is conventionally significant.
* **Probability of Backtest Overfitting** (Bailey, Borwein, López de Prado
  and Zhu, 2017), using combinatorially symmetric cross-validation. The
  common monthly history is split into 16 blocks. For each of the 12,870
  ways to use half the blocks as in-sample, the method picks the best
  in-sample candidate and checks where it ranks in the other half. PBO is
  the share of splits where the in-sample winner falls in the bottom half
  out of sample. Exact-median ranks, possible with an odd number of
  candidates, count as half. Pure noise gives about 0.5. Two candidate sets
  are used:
  * main: 6 candidates including SPY, from December 2006, without the
    factor blend
  * secondary: all 7, from July 2013 (a thin history)

  With so few candidates and months, PBO is a noisy estimate. A single
  pure-noise sample can land anywhere from about 0.1 to 0.9.
* **Block bootstrap.** A circular block bootstrap with 12-month blocks,
  5,000 resamples and a fixed seed. Strategy, SPY and T-bill months are
  resampled together. It gives 95% ranges for the difference in annualized
  return and in Sharpe ratio between each strategy and SPY over the same
  period. A range that includes zero means the difference cannot be told
  apart from luck.

## The report and the verdict

`python -m src.reporting` runs everything and writes one self-contained HTML
file (charts embedded, light and dark versions) plus a short Markdown summary.
It contains:
* the headline metrics table, with SPY over the same dates in each row
* charts: growth of $1 on a log scale, drawdowns, rolling 3-year excess return
  over SPY, calendar-year returns, and monthly allocation
* results by account type and the cost sensitivity at 2, 5 and 10 bp
* the sub-period tables, the downturn analysis and every overfitting check
  explained in plain English
* the French long-history appendix
* the verdict

**Downturns.** Each downturn is SPY's deepest peak-to-recovery decline whose
trough falls in 2000–02, 2007–09, 2020 or 2022 (2022 stays locked).
Strategies are measured from SPY's peak to SPY's recovery:
* **Avoided:** the strategy fell at most half as far as SPY.
* **Whipsawed:** it cut risk exposure by at least 30 percentage points and
  later added it back while SPY was higher than when it sold.

For each strategy the report also lists what it held going into each
downturn and on average from peak to trough.

**Verdict.** The rules were pre-registered in `config/settings.yaml`
(`report.verdict_criteria`) before any real data was seen. A strategy is
*preferable to holding SPY* for an account type only if all of these hold:
* the bootstrap 95% range for its Sharpe advantage is above zero
* its Deflated Sharpe Ratio is at least 0.95
* the main PBO is below 0.5
* its drawdowns are smaller than SPY's in most evaluated windows
* its CAGR is within 1 point a year of SPY's (after tax and liquidated for
  the taxable verdict)

A strategy that passes the drawdown test but fails a statistical test is
reported as *lower drawdowns, but not distinguishable from luck*. Anything
else is *hold SPY*. If nothing qualifies, the report says plainly that
buying and holding an S&P 500 index fund is the better choice.

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
  ETF they need exists and the lookback period is filled. These are the
  first signal dates; trading starts the next day:
  * Sector strategies: January 2000. The sector ETFs launched in December
    1998, and 12-1 momentum needs 14 month-end closes, so the start just
    covers the 2000–2002 bear market.
  * GEM: September 2003. EFA has 12 months of history by 2002, and AGG,
    the defensive holding, only needs to exist (it launched September 2003).
  * GTAA: November 2006, when DBC (launched February 2006) has 10 month-end
    closes for its moving average.
  * Factor blend: July 2013, when QUAL launched.

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

## Paper trading (Alpaca paper endpoint only)

Paper trading uses the official `alpaca-py` client. It only ever talks to
`https://paper-api.alpaca.markets`.

**Setup**
1. Create an Alpaca paper account. Put its keys in `.env` as
   `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`, and set `PAPER_TRADING=true`.
2. Choose the strategy after reviewing the report: set `paper.strategy` in
   `config/settings.yaml`. Until then, `plan` refuses to run. Use one Alpaca
   paper account per strategy.
3. Keep `paper.dry_run: true` until the dry runs look right.

**Monthly cycle.** This mirrors the backtest: signal at the month-end close,
fill at the next trading day's close.

| When (US Eastern) | Command | What it does |
|---|---|---|
| Month-end, after the close | `python -m src.data download` then `python -m src.paper plan` | Computes this month's targets from the latest month-end prices, reads paper positions, saves the intended orders to `logs/paper/plan_*.json`. Sends nothing. |
| Next trading day, before 15:45 | `python -m src.paper submit` | Dry run: prints the orders. Real paper orders need **both** `paper.dry_run: false` **and** `--execute`. Orders are market-on-close in whole shares. |
| Same day, after the close | `python -m src.data download` then `python -m src.paper reconcile` | Records fills in `logs/paper/fills.csv` and compares each fill with that day's close. That close is the price the backtest assumes, so the difference is real slippage in bp (positive = worse). |
| Any time | `python -m src.paper status` | Shows paper equity, cash and positions. |

Everything is logged to `logs/paper_trading.log`.

The plan keeps 0.5% of equity uninvested so whole-share buys can't exceed
cash. It skips trades under $25, but always closes positions the strategy
no longer holds. A month with no new signal leaves positions unchanged.
That applies to buy-and-hold after the first purchase, and to the factor
blend outside December unless drift exceeds 5 points.

**Example cron lines.** Run `crontab -e`. These assume the machine's
timezone is US Eastern and the project lives in `~/cartoon`. Cron can't
target "the last trading day", so the plan job runs on days 28–31 and on
days 1–3. Each run plans from the latest *complete* month-end, so repeats are
harmless, and you review the newest plan before submitting.
```cron
30 17 28-31,1-3 * *  cd ~/cartoon && .venv/bin/python -m src.data download --no-crosscheck && .venv/bin/python -m src.paper plan
00 15 1-4 * *        cd ~/cartoon && .venv/bin/python -m src.paper submit            # dry run unless you add --execute
30 17 1-5 * *        cd ~/cartoon && .venv/bin/python -m src.data download --no-crosscheck && .venv/bin/python -m src.paper reconcile
```
The day-of-week field is deliberately `*`. When both day-of-month and
day-of-week are restricted, cron runs the job if *either* matches, which
would mean every weekday. Weekend runs are harmless: `submit` refuses while
the market is closed.
`submit` refuses to run in four cases:
* the market is closed
* it is past the 15:45 ET cutoff
* it is the signal day itself (orders must fill the next trading day)
* the plan was already submitted

**The held-out exception.** Paper trading needs today's prices, which fall
in the locked held-out period. One narrow, logged path is allowed:
`DataStore.signal_data`. It returns recent prices only for computing this
month's targets and measuring fill slippage, and every use is appended to
`logs/heldout_signal_access.log`. It never computes or reports backtest
performance for the held-out period. The normal analysis path stays locked.

## Safety

* **Paper only.** The paper client refuses to start unless
  `PAPER_TRADING=true` and the configured base URL is exactly the Alpaca
  paper endpoint. It always builds the broker client with `paper=True` and
  the paper URL, then re-checks the URL after construction.
* **No live code.** There is no live-trading code in this project. A test
  scans the source and fails if any non-paper Alpaca host or live setting
  appears.
* **Dry run by default.** Orders are sent only when `paper.dry_run: false`
  **and** `--execute` are both given.
* **No Robinhood.** This project never connects to Robinhood in any form,
  including its Agentic Trading or crypto APIs. A test fails if the source
  mentions it.

**If live trading is ever added later**, it must be a separate module, off by
default, that requires all of these:
1. its own explicitly named environment flag, never `PAPER_TRADING`
2. an interactive typed confirmation on every run
3. a separate config section
4. its own tests

It must not reuse or loosen any of the paper guards above.
