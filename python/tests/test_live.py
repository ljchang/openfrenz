"""The live API, driven by a replayed synthetic recording: no band and no Bluetooth needed."""

import sqlite3
import time

import numpy as np
import pytest

import openfrenz


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "synthetic.sqlite"
    openfrenz.synth(path, seconds=10)
    return path


def wait_until_ended(band, timeout=10.0):
    """Waits until the replay is over and every packet has reached the stream buffers (they are
    filled a moment after the band's own counters move)."""
    deadline = time.monotonic() + timeout
    while True:
        health = band.health()
        buffered = health["streams"]["eeg"]["samples"] == 9 * health["notifications"].get("eeg", 0)
        if not health["connected"] and buffered:
            return
        assert time.monotonic() < deadline, "replay did not finish"
        time.sleep(0.01)


def test_replay_reports_info_and_health(recording):
    with openfrenz.replay(recording, speed=0) as band:
        assert band.is_open
        assert band.info.name == "FRENZ-SYNTH"
        assert band.info.firmware == "synthetic"
        assert band.recording is None
        assert "FRENZ-SYNTH" in repr(band) and "open" in repr(band)
    assert not band.is_open
    # Counters and info stay readable after closing; closing twice is harmless.
    assert band.info.name == "FRENZ-SYNTH"
    assert band.close() == band.health()


def test_replay_records_every_notification(recording, tmp_path, capsys):
    copy = tmp_path / "out" / "copy.sqlite"
    with openfrenz.replay(recording, speed=0, record=copy) as band:
        assert band.recording == copy
        band.mark("hello")
        wait_until_ended(band)
    health = band.health()
    assert "recording to" in capsys.readouterr().out
    assert health["sequence_gaps"] == 0 and health["frames_lost"] == 0
    assert health["notifications"]["eeg"] == 10 * 125 // 9
    assert health["battery_percent"] == 90

    count = lambda db, table: sqlite3.connect(db).execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    assert count(copy, "notifications") == count(recording, "notifications") == sum(health["notifications"].values())
    labels = [row[0] for row in sqlite3.connect(copy).execute("SELECT label FROM markers")]
    assert "hello" in labels


def test_record_into_directory_uses_default_name(recording, tmp_path):
    with openfrenz.replay(recording, speed=0, record=tmp_path) as band:
        pass
    assert band.recording.parent == tmp_path
    assert band.recording.name.startswith("FRENZ-SYNTH-") and band.recording.suffix == ".sqlite"
    assert band.recording.exists()


def test_errors(recording, tmp_path):
    with openfrenz.replay(recording, speed=0) as band:
        # A replay has no device to command.
        with pytest.raises(openfrenz.Disconnected):
            band.battery()
    with pytest.raises(openfrenz.FrenzError, match="closed"):
        band.mark("late")
    with pytest.raises(openfrenz.RecordingError):
        openfrenz.replay(tmp_path / "missing.sqlite")
    with pytest.raises(openfrenz.RecordingError):
        openfrenz.synth(recording)  # never overwrites
    with pytest.raises(ValueError):
        openfrenz.scan(timeout=-1)
    assert issubclass(openfrenz.BluetoothUnavailable, openfrenz.BluetoothError)
    assert issubclass(openfrenz.BandNotFound, openfrenz.FrenzError)


# --- streams -------------------------------------------------------------------------------

EEG_SAMPLES = (10 * 125 // 9) * 9


def read_all(band, name):
    return list(band.stream(name))


def test_stream_delivers_every_sample(recording):
    with openfrenz.replay(recording, speed=0) as band:
        chunks = read_all(band, "eeg")
    data = np.concatenate([c.data for c in chunks], axis=1)
    t = np.concatenate([c.t for c in chunks])
    first = chunks[0]
    assert first.stream == "eeg" and first.channels == openfrenz.SIGNALS["eeg"][1]
    assert data.shape == (4, EEG_SAMPLES) and data.dtype == np.int32
    assert t.dtype == np.float64 and first.device_time.dtype == np.int64
    assert first.duplicate.dtype == np.bool_ and not first.duplicate.any()
    assert all(c.dropped == 0 for c in chunks)
    assert np.all(np.diff(t) > 0), "time must increase within a segment"
    assert set(np.concatenate([c.segment for c in chunks])) == {0}
    assert np.array_equal(np.concatenate([c.index for c in chunks]), np.arange(sum(len(c) for c in chunks)))
    assert data.any()


def test_channels_match_the_session_loader(recording):
    with openfrenz.replay(recording, speed=0) as band:
        wait_until_ended(band)
        for name, (_rate, columns) in openfrenz.SIGNALS.items():
            chunk = band.read(name)
            assert chunk.channels == columns
            assert chunk.data.shape == (len(columns), len(chunk)) and len(chunk) > 0
        assert band.read("battery").data[0, -1] == 90
        assert not band.read("contact").sampled


def test_latest_does_not_consume(recording):
    with openfrenz.replay(recording, speed=0) as band:
        wait_until_ended(band)
        window = band.latest("eeg", seconds=2)
        assert window.data.shape == (4, 250)
        everything = band.read("eeg")
        assert len(everything) == EEG_SAMPLES
        assert np.array_equal(everything.data[:, -250:], window.data)
        assert np.array_equal(everything.t[-250:], window.t)
        assert len(band.latest("accel", seconds=1)) == 50
        # Asking for more than was buffered returns what there is.
        assert len(band.latest("eeg", seconds=3600)) == EEG_SAMPLES


def test_slow_reader_is_told_what_it_missed(recording, tmp_path):
    copy = tmp_path / "copy.sqlite"
    with openfrenz.replay(recording, speed=0, record=copy, buffer_seconds=1) as band:
        wait_until_ended(band)
        chunk = band.read("eeg")
        assert len(chunk) == 125
        assert chunk.dropped == EEG_SAMPLES - 125
        eeg = band.health()["streams"]["eeg"]
        assert eeg["samples"] == EEG_SAMPLES and eeg["dropped"] == chunk.dropped
    # The recording is written independently of the reader and is complete.
    count = lambda db: sqlite3.connect(db).execute("SELECT count(*) FROM notifications").fetchone()[0]
    assert count(copy) == count(recording)


def test_read_timeout_and_end(recording):
    band = openfrenz.replay(recording, speed=1)
    first = band.read("eeg", timeout=5)
    assert first is not None and len(first) % 9 == 0
    # The next packet is about 70 ms away.
    assert band.read("eeg", timeout=0.001) is None
    band.close()
    # What was buffered before closing can still be read; after that the stream has ended.
    with pytest.raises(openfrenz.Disconnected):
        while True:
            band.read("eeg", timeout=1)
    assert read_all(band, "eeg") == []
    with pytest.raises(ValueError, match="unknown stream"):
        band.read("eog")
    with pytest.raises(ValueError):
        band.stream("eog")


def test_live_clock_reports_its_fit(tmp_path):
    # Real pacing, because the live clock assumes the band's real sample rate.
    path = tmp_path / "short.sqlite"
    openfrenz.synth(path, seconds=4)
    with openfrenz.replay(path, speed=1) as band:
        chunks = read_all(band, "eeg")
        eeg = band.health()["streams"]["eeg"]
    t = np.concatenate([c.t for c in chunks])
    arrival = np.concatenate([c.arrival for c in chunks])
    assert eeg["rate_hz"] == pytest.approx(125, rel=0.05) and not eeg["nominal_rate_used"]
    # The synthetic recording delays packets by 4 ms plus up to 120 ms (95th percentile near
    # 100 ms above the fastest); the live fit should report that.
    assert 0.06 < eeg["jitter_p95_s"] < 0.13
    assert np.all(np.diff(t) > 0)
    # A sample is acquired before the packet carrying it arrives, and not long before.
    late = t - arrival
    assert np.mean(late <= 1e-6) > 0.9 and late.max() < 0.02 and late.min() > -0.25


def test_chunk_tables(recording):
    with openfrenz.replay(recording, speed=0) as band:
        wait_until_ended(band)
        eeg, battery = band.read("eeg"), band.read("battery")
    frame = eeg.to_pandas()
    assert list(frame.columns) == [
        "t", "segment", "device_time", "arrival",
        "forehead_right", "comb_left", "forehead_left", "comb_right",
        "duplicate", "held",
    ]  # fmt: skip
    assert len(frame) == len(eeg) and np.array_equal(frame["comb_left"].to_numpy(), eeg.data[1])
    assert list(battery.to_pandas().columns) == ["t", "device_time", "percent"]
    assert "eeg" in repr(eeg) and "4 channels" in repr(eeg)
    try:
        import polars  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="openfrenz\\[polars\\]"):
            eeg.to_polars()
    else:
        table = eeg.to_polars()
        assert table.columns == list(frame.columns) and table.height == len(eeg)


# --- events, commands, export, summary -----------------------------------------------------


def test_events_report_a_gap_and_the_end(recording):
    # Remove one EEG frame from the middle, as if the link had lost it.
    db = sqlite3.connect(recording)
    eeg = [row[0] for row in db.execute("SELECT id FROM notifications WHERE length(payload) = 178 ORDER BY id")]
    db.execute("DELETE FROM notifications WHERE id = ?", (eeg[len(eeg) // 2],))
    db.commit()
    db.close()

    with openfrenz.replay(recording, speed=0) as band:
        events = list(band.events())
        health = band.health()
        index = np.concatenate([c.index for c in read_all(band, "eeg")])
        with pytest.raises(openfrenz.Disconnected):
            band.read_event(timeout=0)
    # The index counts the lost frame: one step of 10 samples, every other step is 1.
    steps = np.diff(index)
    assert sorted(set(steps)) == [1, 10] and (steps == 10).sum() == 1
    kinds = [e.kind for e in events]
    assert kinds.count("sequence_gap") == 1 and kinds[-1] == "disconnected"
    gap = events[kinds.index("sequence_gap")]
    assert gap.detail["frames_lost"] == 1
    assert (gap.detail["received"] - gap.detail["expected"]) % 256 == 1
    assert abs(gap.t - time.time()) < 60 and "sequence_gap" in repr(gap)
    assert health["sequence_gaps"] == 1 and health["frames_lost"] == 1 and health["events_dropped"] == 0
    # The exporter keeps the segment and leaves the lost samples out; so do the live buffers.
    assert health["streams"]["eeg"]["samples"] == EEG_SAMPLES - 9
    assert health["streams"]["eeg"]["segment"] == 0


def test_slow_reader_gets_a_lagged_event(recording):
    with openfrenz.replay(recording, speed=0, buffer_seconds=1) as band:
        wait_until_ended(band)
        assert band.read_event(timeout=0).kind == "disconnected"
        chunk = band.read("eeg")
        event = band.read_event(timeout=1)
    assert event.kind == "lagged"
    assert event.detail == {"stream": "eeg", "dropped": chunk.dropped}


def test_commands_are_checked_before_sending(recording):
    assert openfrenz.COMMANDS["stream_control"] == ("mask", "enable")
    with openfrenz.replay(recording, speed=0) as band:
        with pytest.raises(ValueError, match="unknown command"):
            band.send("self_destruct")
        for bad in [("stream_control",), ("stream_control", 255), ("get_battery", 1), ("set_prompt_volume", 0, 1)]:
            with pytest.raises(TypeError, match="takes"):
                band.send(*bad)
        with pytest.raises(TypeError):
            band.send("stream_control", 255, mask=255)
        with pytest.raises(TypeError):
            band.request("heart_rate_periodic", on=True)
        with pytest.raises(OverflowError):
            band.send("set_prompt_volume", 300)
        # Well-formed commands reach the link, which a replay does not have.
        for good in [("get_serial_number",), ("time_sync",), ("stream_control", 255, True)]:
            with pytest.raises(openfrenz.Disconnected):
                band.request(*good)
        with pytest.raises(openfrenz.Disconnected):
            band.send("stream_control", mask=255, enable=False)


def test_export_matches_what_was_streamed(recording, tmp_path):
    with openfrenz.replay(recording, speed=0, record=tmp_path / "copy.sqlite") as band:
        live = {name: read_all(band, name) for name in openfrenz.SIGNALS}
    session = openfrenz.export(recording, tmp_path / "sessions")
    assert session.path == tmp_path / "sessions" / "synthetic"
    for name, chunks in live.items():
        table = session[name].sort_values(["packet", "sample"])
        streamed = np.concatenate([c.data for c in chunks], axis=1)
        assert np.array_equal(streamed, table[list(openfrenz.SIGNALS[name][1])].to_numpy().T)
        assert np.array_equal(np.concatenate([c.device_time for c in chunks]), table["device_time"].to_numpy())
        assert set(chunks[0].to_pandas().columns) <= set(table.columns)
    # Exporting is deterministic, and a recording of the replay exports to the same samples.
    first = {p.name: p.read_bytes() for p in session.path.iterdir()}
    again = openfrenz.export(recording, tmp_path / "sessions")
    assert first == {p.name: p.read_bytes() for p in again.path.iterdir()}
    copy = openfrenz.export(band.recording, tmp_path / "sessions")
    assert np.array_equal(copy.signal("eeg")[1], session.signal("eeg")[1])
    with pytest.raises(openfrenz.RecordingError):
        openfrenz.export(tmp_path / "missing.sqlite", tmp_path / "sessions")


def test_summary(recording, tmp_path):
    summary = openfrenz.summary(recording)
    assert summary["seconds"] == pytest.approx(10, abs=0.5)
    assert summary["samples"]["eeg"] == EEG_SAMPLES and set(summary["samples"]) == set(openfrenz.SIGNALS)
    assert summary["rates_hz"]["eeg"] == pytest.approx(125, rel=0.05)
    assert summary["health"]["sequence_gaps"] == 0 and "connected" not in summary["health"]
    with pytest.raises(openfrenz.RecordingError):
        openfrenz.summary(tmp_path / "missing.sqlite")
