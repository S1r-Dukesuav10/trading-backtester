"""Tests for the Dukascopy downloader. No network: subprocess is faked."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.data.dukascopy import (
    DukascopyDownloader,
    DukascopyError,
    download_range,
    format_report,
    has_data_rows,
    head_rows,
    merge_csvs,
    tail_rows,
)

CSV_HEADER = "timestamp,open,high,low,close,volume"


class FakeDukascopyCLI:
    """Stands in for `npx dukascopy-node ...`; writes a fake CSV per call."""

    def __init__(
        self,
        *,
        fail_months: set[str] | None = None,
        empty_months: set[str] | None = None,
        split_months: set[str] | None = None,
    ) -> None:
        self.fail_months = set(fail_months or ())
        self.empty_months = set(empty_months or ())
        self.split_months = set(split_months or ())
        self.commands: list[list[str]] = []

    def month_of(self, cmd: list[str]) -> str:
        date_from = cmd[cmd.index("-from") + 1]
        return date_from[:7]

    def out_dir(self, cmd: list[str]) -> Path:
        return Path(cmd[cmd.index("-dir") + 1])

    def __call__(
        self, cmd: list[str], *, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(cmd))
        month = self.month_of(cmd)
        out_dir = self.out_dir(cmd)

        if month in self.fail_months:
            return subprocess.CompletedProcess(cmd, 1, "", "simulated failure")
        if month in self.empty_months:
            (out_dir / f"{month}.csv").write_text(CSV_HEADER + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        rows = [
            f"2024-01-0{index + 1}T00:00:00.000Z,2000.{index},2001.{index},"
            f"1999.{index},2000.{index + 1},10"
            for index in range(3)
        ]
        if month in self.split_months:
            (out_dir / f"{month}-part1.csv").write_text(
                "\n".join([CSV_HEADER, *rows[:2]]) + "\n", encoding="utf-8"
            )
            (out_dir / f"{month}-part2.csv").write_text(
                "\n".join([CSV_HEADER, *rows[2:]]) + "\n", encoding="utf-8"
            )
        else:
            (out_dir / "nested" / f"{month}.csv").parent.mkdir(exist_ok=True)
            (out_dir / "nested" / f"{month}.csv").write_text(
                "\n".join([CSV_HEADER, *rows]) + "\n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(cmd, 0, "ok", "")


def make_downloader(tmp_path, cli, **kwargs) -> DukascopyDownloader:
    return DukascopyDownloader(tmp_path, run=cli, sleep=lambda _seconds: None, **kwargs)


# -- paths and command construction ---------------------------------------


def test_month_path_layout(tmp_path):
    downloader = DukascopyDownloader(tmp_path)
    assert downloader.month_path("2024-01") == (
        tmp_path / "XAUUSD" / "XAUUSD-m1-bid-2024-01.csv"
    )


def test_command_uses_one_month_per_invocation(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.dukascopy.resolve_npx", lambda: "npx")
    downloader = DukascopyDownloader(tmp_path)
    cmd = downloader.build_command("2024-02", tmp_path / "scratch")

    assert cmd[0] == "npx"
    assert cmd[1:3] == ["-y", "dukascopy-node"]
    assert "xauusd" in cmd  # lower-cased instrument id
    assert "2024-02-01" in cmd and "2024-02-29" in cmd  # leap-year month end
    assert "m1" in cmd
    assert "bid" in cmd
    assert "csv" in cmd
    assert str(tmp_path / "scratch") in cmd
    # Explicit UTC offset, never the machine's local time.
    assert cmd[cmd.index("-utc") + 1] == "0"
    # Output dir flag is -dir (-d would be --debug).
    assert cmd[cmd.index("-dir") + 1] == str(tmp_path / "scratch")
    assert "-d" not in cmd
    # Volumes on by default so the canonical schema has real volume data.
    assert "-v" in cmd


def test_command_omits_volumes_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr("app.data.dukascopy.resolve_npx", lambda: "npx")
    downloader = DukascopyDownloader(tmp_path, volumes=False)
    cmd = downloader.build_command("2024-01", tmp_path / "scratch")
    assert "-v" not in cmd


def test_missing_npx_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("app.data.dukascopy.shutil.which", lambda _name: None)
    downloader = DukascopyDownloader(tmp_path)
    with pytest.raises(DukascopyError, match="npx"):
        downloader.build_command("2024-01", tmp_path / "scratch")


def test_invalid_side_rejected(tmp_path):
    with pytest.raises(ValueError, match="side"):
        DukascopyDownloader(tmp_path, side="mid")


# -- downloading ----------------------------------------------------------


def test_downloads_month_to_canonical_filename(tmp_path):
    cli = FakeDukascopyCLI()
    downloader = make_downloader(tmp_path, cli)

    report = downloader.download_months(["2024-01"])

    assert report.rows[0].status == "downloaded"
    target = tmp_path / "XAUUSD" / "XAUUSD-m1-bid-2024-01.csv"
    assert target.exists()
    assert has_data_rows(target)
    assert report.total_size_bytes == target.stat().st_size
    # Nested tool output is moved to the canonical flat name.
    assert not (tmp_path / "XAUUSD" / "nested").exists()


def test_skips_existing_non_empty_file(tmp_path):
    cli = FakeDukascopyCLI()
    downloader = make_downloader(tmp_path, cli)
    downloader.download_months(["2024-01", "2024-02"])
    assert len(cli.commands) == 2

    # Second run: both months already on disk -> no subprocess at all.
    cli.commands.clear()
    report = downloader.download_months(["2024-01", "2024-02"])

    assert cli.commands == []
    assert [r.status for r in report.rows] == ["cached", "cached"]


def test_existing_empty_file_is_retried(tmp_path):
    cli = FakeDukascopyCLI()
    downloader = make_downloader(tmp_path, cli)
    target = downloader.month_path("2024-01")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")

    report = downloader.download_months(["2024-01"])

    assert len(cli.commands) == 1
    assert report.rows[0].status == "downloaded"
    assert has_data_rows(target)


def test_empty_month_reported_and_file_removed(tmp_path):
    cli = FakeDukascopyCLI(empty_months={"2024-03"})
    downloader = make_downloader(tmp_path, cli)

    report = downloader.download_months(["2024-03"])

    assert report.rows[0].status == "empty"
    assert report.empty and not report.ok
    assert not downloader.month_path("2024-03").exists()  # re-run will retry


def test_empty_month_quotes_tool_output(tmp_path):
    class NoisyEmpty(FakeDukascopyCLI):
        """Emulates the real tool: exit 0, claims success, writes 0 bytes."""

        def __call__(self, cmd, *, timeout):
            self.commands.append(list(cmd))
            target = self.out_dir(cmd) / "xauusd-m1-bid-2024-01.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
            return subprocess.CompletedProcess(
                cmd, 0, "√ File saved: xauusd-m1-bid-2024-01.csv (0 Bytes)", ""
            )

    downloader = DukascopyDownloader(tmp_path, run=NoisyEmpty(), sleep=lambda _s: None)
    report = downloader.download_months(["2024-01"])

    assert report.rows[0].status == "empty"
    assert "0 Bytes" in (report.rows[0].error or "")
    assert "0 Bytes" in format_report(report, root=tmp_path)


def test_aborts_after_consecutive_empty_months(tmp_path):
    months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
    cli = FakeDukascopyCLI(empty_months=set(months))
    downloader = make_downloader(tmp_path, cli)

    report = downloader.download_months(months)

    statuses = [r.status for r in report.rows]
    assert statuses == ["empty", "empty", "empty", "skipped", "skipped"]
    # Only the three months up to the abort threshold were attempted.
    assert len(cli.commands) == 3
    assert "aborted early" in (report.skipped[0].error or "")
    assert not report.ok


def test_empty_run_does_not_abort_when_disabled(tmp_path):
    months = ["2024-01", "2024-02", "2024-03", "2024-04"]
    cli = FakeDukascopyCLI(empty_months=set(months))
    downloader = make_downloader(tmp_path, cli, abort_after_empty_months=0)

    report = downloader.download_months(months)

    assert [r.status for r in report.rows] == ["empty"] * 4
    assert len(cli.commands) == 4


def test_successful_month_resets_the_empty_counter(tmp_path):
    months = ["2024-01", "2024-02", "2024-03"]
    cli = FakeDukascopyCLI(empty_months={"2024-01", "2024-03"})
    downloader = make_downloader(tmp_path, cli)

    report = downloader.download_months(months)

    assert [r.status for r in report.rows] == ["empty", "downloaded", "empty"]
    assert not report.skipped  # never three in a row


def test_failed_month_retried_then_succeeds(tmp_path):
    class FlakyOnce(FakeDukascopyCLI):
        calls = 0

        def __call__(self, cmd, *, timeout):
            FlakyOnce.calls += 1
            if FlakyOnce.calls == 1:
                self.commands.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 1, "", "flaky")
            return super().__call__(cmd, timeout=timeout)

    cli = FlakyOnce()
    waits: list[float] = []
    downloader = DukascopyDownloader(
        tmp_path, run=cli, sleep=waits.append, retry_wait_seconds=7.0
    )

    report = downloader.download_months(["2024-01"])

    assert report.rows[0].status == "downloaded"
    assert report.rows[0].attempts == 2
    assert waits == [7.0]  # one retry wait, no inter-month delay


def test_gives_up_after_max_attempts(tmp_path):
    cli = FakeDukascopyCLI(fail_months={"2024-05"})
    waits: list[float] = []
    downloader = DukascopyDownloader(
        tmp_path,
        run=cli,
        sleep=waits.append,
        max_attempts=3,
        retry_wait_seconds=5.0,
    )

    report = downloader.download_months(["2024-05"])

    assert report.rows[0].status == "failed"
    assert report.rows[0].attempts == 3
    assert len(cli.commands) == 3
    assert "simulated failure" in (report.rows[0].error or "")
    assert waits == [5.0, 10.0]  # escalating backoff between attempts
    assert not report.ok


def test_no_csv_produced_is_a_failure(tmp_path):
    def runner(cmd, *, timeout):
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    downloader = DukascopyDownloader(
        tmp_path, run=runner, sleep=lambda _s: None, max_attempts=2
    )
    report = downloader.download_months(["2024-01"])

    assert report.rows[0].status == "failed"
    assert "no CSV" in (report.rows[0].error or "")


def test_runner_exception_is_captured(tmp_path):
    def runner(cmd, *, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    downloader = DukascopyDownloader(
        tmp_path, run=runner, sleep=lambda _s: None, max_attempts=2
    )
    report = downloader.download_months(["2024-01"])

    assert report.rows[0].status == "failed"
    assert "TimeoutExpired" in (report.rows[0].error or "")


def test_split_output_is_merged_with_single_header(tmp_path):
    cli = FakeDukascopyCLI(split_months={"2024-01"})
    downloader = make_downloader(tmp_path, cli)

    report = downloader.download_months(["2024-01"])

    assert report.rows[0].status == "downloaded"
    text = downloader.month_path("2024-01").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0] == CSV_HEADER
    assert sum(1 for line in lines if line == CSV_HEADER) == 1
    assert len(lines) == 4  # header + 3 data rows


def test_delay_between_months_only(tmp_path):
    cli = FakeDukascopyCLI()
    waits: list[float] = []
    downloader = DukascopyDownloader(
        tmp_path, run=cli, sleep=waits.append, delay_seconds=2.5
    )

    downloader.download_months(["2024-01", "2024-02", "2024-03"])

    assert waits == [2.5, 2.5]  # between months, never after the last


def test_download_range_builds_inclusive_months(tmp_path):
    cli = FakeDukascopyCLI()
    report = download_range(
        "2024-01", "2024-03", root=tmp_path, run=cli, sleep=lambda _s: None
    )

    assert [r.month for r in report.rows] == ["2024-01", "2024-02", "2024-03"]
    assert all(r.status == "downloaded" for r in report.rows)
    assert report.ok


def test_reversed_range_raises(tmp_path):
    with pytest.raises(ValueError, match="before start"):
        download_range("2024-05", "2024-01", root=tmp_path)


def test_timeout_is_passed_to_runner(tmp_path):
    seen: dict[str, float] = {}

    def runner(cmd, *, timeout):
        seen["timeout"] = timeout
        out_dir = Path(cmd[cmd.index("-d") + 1])
        (out_dir / "x.csv").write_text(f"{CSV_HEADER}\n0,1,1,0,1,1\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    downloader = DukascopyDownloader(
        tmp_path, run=runner, sleep=lambda _s: None, timeout=123.0
    )
    downloader.download_months(["2024-01"])

    assert seen["timeout"] == 123.0


# -- helpers and reporting ------------------------------------------------


def test_has_data_rows(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    header_only = tmp_path / "header.csv"
    header_only.write_text(CSV_HEADER + "\n", encoding="utf-8")
    data = tmp_path / "data.csv"
    data.write_text(f"{CSV_HEADER}\n0,1,1,0,1,1\n", encoding="utf-8")
    headerless = tmp_path / "headerless.csv"
    headerless.write_text("0,1,1,0,1,1\n", encoding="utf-8")

    assert not has_data_rows(empty)
    assert not has_data_rows(header_only)
    assert has_data_rows(data)
    assert has_data_rows(headerless)
    assert not has_data_rows(tmp_path / "missing.csv")


def test_merge_csvs_keeps_one_header(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text(f"{CSV_HEADER}\n1,1,1,1,1,1\n", encoding="utf-8")
    b = tmp_path / "b.csv"
    b.write_text(f"{CSV_HEADER}\n2,2,2,2,2,2\n", encoding="utf-8")
    target = tmp_path / "merged.csv"

    merge_csvs([a, b], target)

    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines == [CSV_HEADER, "1,1,1,1,1,1", "2,2,2,2,2,2"]


def test_head_and_tail_rows(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text(f"{CSV_HEADER}\n1\n2\n3\n4\n5\n", encoding="utf-8")

    assert head_rows(path, 2) == [CSV_HEADER, "1"]
    assert tail_rows(path, 2) == ["4", "5"]


def test_format_report_lists_counts_rows_and_failures(tmp_path):
    cli = FakeDukascopyCLI(fail_months={"2024-03"})
    downloader = make_downloader(tmp_path, cli)

    report = downloader.download_months(["2024-01", "2024-02", "2024-03"])
    text = format_report(report, root=tmp_path)

    assert "3 month(s) requested" in text
    assert "downloaded: 2" in text
    assert "failed: 1" in text
    assert "range on disk: 2024-01 .. 2024-02" in text
    assert "first 5 rows of XAUUSD-m1-bid-2024-01.csv" in text
    assert "last 3 rows of XAUUSD-m1-bid-2024-02.csv" in text
    assert "FAILED 2024-03" in text
