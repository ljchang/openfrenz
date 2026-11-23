//! Python bindings for the FRENZ brainband client, built into the `openfrenz` wheel as
//! `openfrenz._native`.
//!
//! Bluetooth, decoding and recording run on a Tokio runtime owned by this module. Python threads
//! only wait on it, without holding the interpreter lock, and are interrupted by Ctrl-C.

mod streams;

use std::future::Future;
use std::path::{MAIN_SEPARATOR, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Duration;

use frenz::proto::{Command, ResponseBody};
use frenz::{ConnectOptions, Health, Recorder, ReplayOptions, StartSequence};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyTuple};
use tokio::task::JoinHandle;

use crate::streams::{Detail, End, Note, Streams};
use tokio::runtime::Runtime;

create_exception!(openfrenz, FrenzError, PyException, "Base class for openfrenz errors.");
create_exception!(
    openfrenz,
    BandNotFound,
    FrenzError,
    "No band with this name was seen within the scan timeout."
);
create_exception!(
    openfrenz,
    BluetoothError,
    FrenzError,
    "The Bluetooth stack reported an error."
);
create_exception!(
    openfrenz,
    BluetoothUnavailable,
    BluetoothError,
    "No Bluetooth adapter is available, or the operating system refused access to it."
);
create_exception!(
    openfrenz,
    CommandTimeout,
    FrenzError,
    "The band did not answer a command in time."
);
create_exception!(
    openfrenz,
    CommandFailed,
    FrenzError,
    "The band answered a command with an error status."
);
create_exception!(
    openfrenz,
    Disconnected,
    FrenzError,
    "The connection to the band is down."
);
create_exception!(
    openfrenz,
    RecordingError,
    FrenzError,
    "Reading or writing a recording file failed."
);

fn to_py(error: frenz::Error) -> PyErr {
    let message = error.to_string();
    match error {
        frenz::Error::NotFound { .. } => BandNotFound::new_err(message),
        frenz::Error::NoAdapter => BluetoothUnavailable::new_err(format!(
            "{message}. On macOS, allow Bluetooth for the application running Python in System Settings → \
             Privacy & Security → Bluetooth."
        )),
        frenz::Error::Bluetooth(_) => BluetoothError::new_err(message),
        frenz::Error::Timeout { .. } => CommandTimeout::new_err(message),
        frenz::Error::CommandFailed { .. } => CommandFailed::new_err(message),
        frenz::Error::Disconnected => Disconnected::new_err(message),
        frenz::Error::Storage(_) | frenz::Error::Io(_) => RecordingError::new_err(message),
        frenz::Error::MissingCharacteristic(_) => FrenzError::new_err(message),
    }
}

fn runtime() -> PyResult<&'static Runtime> {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    if let Some(runtime) = RUNTIME.get() {
        return Ok(runtime);
    }
    let built = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("openfrenz")
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("could not start the openfrenz runtime: {e}")))?;
    Ok(RUNTIME.get_or_init(|| built))
}

/// Runs `future` to completion without holding the interpreter lock, checking for Ctrl-C ten
/// times a second. On Ctrl-C the future is dropped and `KeyboardInterrupt` is raised.
fn wait<T: Send>(py: Python<'_>, future: impl Future<Output = T> + Send) -> PyResult<T> {
    let runtime = runtime()?;
    let mut future = std::pin::pin!(future);
    loop {
        let step = py.detach(|| {
            runtime.block_on(async { tokio::time::timeout(Duration::from_millis(100), future.as_mut()).await })
        });
        match step {
            Ok(value) => return Ok(value),
            Err(_) => py.check_signals()?,
        }
    }
}

fn seconds(value: f64, what: &str) -> PyResult<Duration> {
    Duration::try_from_secs_f64(value)
        .map_err(|_| PyValueError::new_err(format!("{what} must be a non-negative number of seconds, got {value}")))
}

/// Where to record: `True` → `recordings/NAME-UNIXTIME.sqlite`, `False` → nowhere, a directory →
/// that directory with the default file name, anything else → that file.
fn recording_path(record: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<PathBuf>> {
    let default_name = || {
        let unix = frenz::clock::wall_ns() / 1_000_000_000;
        format!("{name}-{unix}.sqlite")
    };
    if let Ok(flag) = record.cast::<PyBool>() {
        return Ok(flag.is_true().then(|| PathBuf::from("recordings").join(default_name())));
    }
    let path: PathBuf = record
        .extract()
        .map_err(|_| PyValueError::new_err("record must be True, False, or a file or directory path"))?;
    let names_directory = path.is_dir() || path.to_string_lossy().ends_with(['/', MAIN_SEPARATOR]);
    Ok(Some(if names_directory {
        path.join(default_name())
    } else {
        path
    }))
}

fn start_recorder(py: Python<'_>, path: Option<PathBuf>, band: &frenz::Band) -> PyResult<Option<(PathBuf, Recorder)>> {
    let Some(path) = path else { return Ok(None) };
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| to_py(e.into()))?;
    }
    // The recorder spawns onto the runtime, so create it from inside.
    let _inside = runtime()?.enter();
    let recorder = Recorder::create(&path, band).map_err(to_py)?;
    let print = py.import("builtins")?.getattr("print")?;
    print.call1((format!("openfrenz: recording to {}", path.display()),))?;
    Ok(Some((path, recorder)))
}

/// The band's own counters, as a dict.
fn counters_dict<'py>(py: Python<'py>, health: &Health) -> PyResult<Bound<'py, PyDict>> {
    let counts = |map: &std::collections::BTreeMap<&'static str, u64>| -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, count) in map {
            dict.set_item(name, count)?;
        }
        Ok(dict)
    };
    let dict = PyDict::new(py);
    dict.set_item("connected", health.connected)?;
    dict.set_item("notifications", counts(&health.notifications)?)?;
    dict.set_item("decode_errors", counts(&health.decode_errors)?)?;
    dict.set_item("irregular_eeg_frames", health.irregular_eeg_frames)?;
    dict.set_item("sequence_gaps", health.sequence_gaps)?;
    dict.set_item("frames_lost", health.frames_lost)?;
    dict.set_item("counter_restarts", health.counter_restarts)?;
    dict.set_item("reconnects", health.reconnects)?;
    dict.set_item("battery_percent", health.battery_percent)?;
    dict.set_item("contact_state", health.contact_state)?;
    Ok(dict)
}

fn health_dict<'py>(py: Python<'py>, health: &Health, streams: &Streams) -> PyResult<Bound<'py, PyDict>> {
    let dict = counters_dict(py, health)?;
    dict.set_item("events_missed", streams.events_missed())?;
    dict.set_item("events_dropped", streams.events_dropped())?;
    let per_stream = PyDict::new(py);
    for stream in streams.health() {
        let entry = PyDict::new(py);
        entry.set_item("samples", stream.samples)?;
        entry.set_item("buffered", stream.buffered)?;
        entry.set_item("dropped", stream.dropped)?;
        entry.set_item("inactive_nonzero", stream.inactive_nonzero)?;
        entry.set_item("segment", stream.segment)?;
        if let Some(fit) = stream.fit {
            entry.set_item("rate_hz", 1.0 / fit.period)?;
            entry.set_item("jitter_p50_s", fit.jitter_p50)?;
            entry.set_item("jitter_p95_s", fit.jitter_p95)?;
            entry.set_item("nominal_rate_used", fit.nominal)?;
        }
        per_stream.set_item(stream.name, entry)?;
    }
    dict.set_item("streams", per_stream)?;
    Ok(dict)
}

fn column<const N: usize>(py: Python<'_>, values: impl Iterator<Item = [u8; N]>) -> Bound<'_, PyByteArray> {
    let bytes: Vec<u8> = values.flatten().collect();
    PyByteArray::new(py, &bytes)
}

/// Builds an `openfrenz.Chunk` from buffered rows. The arrays are handed over as raw bytes and
/// wrapped by NumPy on the Python side, so this crate needs no NumPy build dependency.
fn chunk_object(py: Python<'_>, chunk: &streams::Chunk) -> PyResult<Py<PyAny>> {
    let rows = &chunk.rows;
    // (channels, n) in C order: one channel after another.
    let data = column(
        py,
        (0..chunk.channels.len()).flat_map(|c| {
            rows.iter()
                .map(move |r| r.values.get(c).copied().unwrap_or(0).to_ne_bytes())
        }),
    );
    let built = py.import("openfrenz.chunk")?.getattr("Chunk")?.call_method1(
        "_from_buffers",
        (
            chunk.stream,
            chunk.channels.to_vec(),
            chunk.sampled,
            data,
            column(py, rows.iter().map(|r| r.t.to_ne_bytes())),
            column(py, rows.iter().map(|r| r.arrival.to_ne_bytes())),
            column(py, rows.iter().map(|r| r.device_time.to_ne_bytes())),
            column(py, rows.iter().map(|r| r.segment.to_ne_bytes())),
            column(py, rows.iter().map(|r| [u8::from(r.duplicate)])),
            column(py, rows.iter().map(|r| [u8::from(r.held)])),
            chunk.dropped,
        ),
    )?;
    Ok(built.unbind())
}

/// Command names and their arguments, for error messages and `openfrenz.COMMANDS`.
const COMMANDS: [(&str, &[&str]); 14] = [
    ("get_firmware_version", &[]),
    ("get_battery", &[]),
    ("get_serial_number", &[]),
    ("get_device_name", &[]),
    ("get_device_time", &[]),
    ("get_device_id", &[]),
    ("get_imu_calibration", &[]),
    ("time_sync", &["unix_seconds"]),
    ("heart_rate_periodic", &["enable"]),
    ("stream_control", &["mask", "enable"]),
    ("set_prompt_volume", &["volume"]),
    ("on_connected", &[]),
    ("start_session", &[]),
    ("end_session", &[]),
];

/// Turns `("stream_control", 0xff, True)` or `("stream_control", mask=0xff, enable=True)` into a
/// command.
fn parse_command(name: &str, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Command> {
    let Some((_, parameters)) = COMMANDS.iter().find(|(known, _)| *known == name) else {
        let known: Vec<_> = COMMANDS.iter().map(|(known, _)| *known).collect();
        return Err(PyValueError::new_err(format!(
            "unknown command {name:?}; commands are {}",
            known.join(", ")
        )));
    };
    let usage = || PyTypeError::new_err(format!("{name} takes ({})", parameters.join(", ")));
    if args.len() > parameters.len() {
        return Err(usage());
    }
    if let Some(kwargs) = kwargs {
        for key in kwargs.keys() {
            let key: String = key.extract()?;
            let position = parameters.iter().position(|p| *p == key).ok_or_else(usage)?;
            if position < args.len() {
                return Err(usage());
            }
        }
    }
    let value = |position: usize| -> PyResult<Option<Bound<'_, PyAny>>> {
        if position < args.len() {
            return args.get_item(position).map(Some);
        }
        match (kwargs, parameters.get(position)) {
            (Some(kwargs), Some(key)) => kwargs.get_item(key),
            _ => Ok(None),
        }
    };
    let required = |position: usize| value(position)?.ok_or_else(usage);
    Ok(match name {
        "get_firmware_version" => Command::GetFirmwareVersion,
        "get_battery" => Command::GetBattery,
        "get_serial_number" => Command::GetSerialNumber,
        "get_device_name" => Command::GetDeviceName,
        "get_device_time" => Command::GetDeviceTime,
        "get_device_id" => Command::GetDeviceId,
        "get_imu_calibration" => Command::GetImuCalibration,
        "on_connected" => Command::OnConnected,
        "start_session" => Command::StartSession,
        "end_session" => Command::EndSession,
        // Without an argument, the computer's current time.
        "time_sync" => Command::TimeSync(match value(0)? {
            Some(unix) => unix.extract()?,
            None => u32::try_from(frenz::clock::wall_ns() / 1_000_000_000).unwrap_or(u32::MAX),
        }),
        "heart_rate_periodic" => Command::HeartRatePeriodic(required(0)?.extract()?),
        "stream_control" => Command::StreamControl {
            mask: required(0)?.extract()?,
            enable: required(1)?.extract()?,
        },
        "set_prompt_volume" => Command::SetPromptVolume(required(0)?.extract()?),
        _ => return Err(usage()),
    })
}

/// The band's answer to a command.
#[pyclass(module = "openfrenz", frozen)]
struct Response {
    /// Opcode of the command answered.
    #[pyo3(get)]
    opcode: u8,
    /// Status byte; 0 means success.
    #[pyo3(get)]
    status: u8,
    payload: Vec<u8>,
    value: Py<PyAny>,
}

impl Response {
    fn new(py: Python<'_>, response: &frenz::proto::Response) -> PyResult<Self> {
        let payload = response.payload().to_vec();
        let raw = || PyBytes::new(py, &payload).into_any().unbind();
        let text = |text: &str| -> PyResult<Py<PyAny>> { Ok(text.into_pyobject(py)?.into_any().unbind()) };
        let value = match response.body() {
            Ok(ResponseBody::Ack) => py.None(),
            Ok(
                ResponseBody::FirmwareVersion(t)
                | ResponseBody::SerialNumber(t)
                | ResponseBody::DeviceName(t)
                | ResponseBody::DeviceId(t),
            ) => text(t)?,
            Ok(ResponseBody::Battery(percent)) => percent.into_pyobject(py)?.into_any().unbind(),
            Ok(ResponseBody::DeviceTime(unix)) => unix.into_pyobject(py)?.into_any().unbind(),
            Ok(ResponseBody::ImuCalibration { std, diff }) => {
                let dict = PyDict::new(py);
                dict.set_item("std", (std[0], std[1], std[2]))?;
                dict.set_item("diff", (diff[0], diff[1], diff[2]))?;
                dict.into_any().unbind()
            }
            Ok(ResponseBody::Other(_)) | Err(_) => raw(),
        };
        Ok(Self {
            opcode: response.opcode,
            status: response.status,
            payload,
            value,
        })
    }
}

#[pymethods]
impl Response {
    /// Bytes after the opcode and status.
    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.payload)
    }

    /// The payload interpreted: text, a number, a dict for the IMU calibration report, None for a
    /// bare acknowledgement, or the raw bytes when the layout is not known.
    #[getter]
    fn value(&self, py: Python<'_>) -> Py<PyAny> {
        self.value.clone_ref(py)
    }

    /// True when the status byte reports success.
    #[getter]
    fn ok(&self) -> bool {
        self.status == 0
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Response(opcode=0x{:02x}, status={}, value={})",
            self.opcode,
            self.status,
            self.value.bind(py).repr()?
        ))
    }
}

/// Something that happened on the connection.
#[pyclass(module = "openfrenz", name = "Event", frozen)]
struct BandEvent {
    /// What happened: `sequence_gap`, `counter_restart`, `off_head`, `on_head`,
    /// `power_off_warning`, `disconnected`, `reconnecting`, `reconnected`, `reconnect_failed`,
    /// `decode_error`, `lagged` or `events_missed`.
    #[pyo3(get)]
    kind: &'static str,
    /// When, in Unix seconds.
    #[pyo3(get)]
    t: f64,
    detail: Py<PyDict>,
}

impl BandEvent {
    fn new(py: Python<'_>, note: &Note) -> PyResult<Self> {
        let detail = PyDict::new(py);
        for (key, value) in &note.detail {
            match value {
                Detail::Int(number) => detail.set_item(key, number)?,
                Detail::Text(text) => detail.set_item(key, text)?,
            }
        }
        Ok(Self {
            kind: note.kind,
            t: note.t,
            detail: detail.unbind(),
        })
    }
}

#[pymethods]
impl BandEvent {
    /// Further facts about the event, by name (empty for most kinds).
    #[getter]
    fn detail(&self, py: Python<'_>) -> Py<PyDict> {
        self.detail.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Event({:?}, t={:.3}, {})",
            self.kind,
            self.t,
            self.detail.bind(py).repr()?
        ))
    }
}

fn stream_index(name: &str) -> PyResult<usize> {
    streams::index(name).ok_or_else(|| {
        let known: Vec<_> = streams::names().collect();
        PyValueError::new_err(format!("unknown stream {name:?}; streams are {}", known.join(", ")))
    })
}

/// A band seen while scanning.
#[pyclass(module = "openfrenz", frozen, get_all)]
struct Discovered {
    /// Advertised name; pass it to `connect`.
    name: String,
    /// Signal strength at discovery (dBm), if reported.
    rssi: Option<i16>,
}

#[pymethods]
impl Discovered {
    fn __repr__(&self) -> String {
        match self.rssi {
            Some(rssi) => format!("Discovered(name={:?}, rssi={rssi})", self.name),
            None => format!("Discovered(name={:?}, rssi=None)", self.name),
        }
    }
}

/// IMU calibration report as Python sees it: `(std per axis, diff per axis)`.
type ImuCalibration = ((u8, u8, u8), (u16, u16, u16));

/// Facts about a connected band.
#[pyclass(module = "openfrenz", frozen, get_all)]
#[derive(Clone)]
struct BandInfo {
    /// Advertised name.
    name: String,
    /// Firmware version, if the band reported one.
    firmware: Option<String>,
    /// IMU calibration report `((std_x, std_y, std_z), (diff_x, diff_y, diff_z))`, if reported.
    imu_calibration: Option<ImuCalibration>,
}

impl From<frenz::BandInfo> for BandInfo {
    fn from(info: frenz::BandInfo) -> Self {
        Self {
            name: info.name,
            firmware: info.firmware,
            imu_calibration: info
                .imu_calibration
                .map(|([a, b, c], [d, e, f])| ((a, b, c), (d, e, f))),
        }
    }
}

#[pymethods]
impl BandInfo {
    fn __repr__(&self) -> String {
        format!(
            "BandInfo(name={:?}, firmware={:?}, imu_calibration={:?})",
            self.name, self.firmware, self.imu_calibration
        )
    }
}

struct Open {
    band: Arc<frenz::Band>,
    recorder: Option<Recorder>,
}

struct Closed {
    info: BandInfo,
    health: Health,
}

enum State {
    Open(Open),
    Closed(Closed),
}

/// A connected band (or a recording being replayed). Use it in a `with` block, or call `close()`.
#[pyclass(module = "openfrenz", frozen)]
struct Band {
    state: Mutex<State>,
    recording: Option<PathBuf>,
    /// Sample buffers. They outlive `close()`, so what was buffered can still be read afterwards.
    streams: Arc<Streams>,
    pump: JoinHandle<()>,
}

/// How a connection feeds its buffers.
struct Feed {
    events: tokio::sync::broadcast::Receiver<frenz::Event>,
    buffer_seconds: f64,
    reconnects: bool,
    end: End,
}

impl Band {
    fn open(band: frenz::Band, recorder: Option<(PathBuf, Recorder)>, feed: Feed) -> PyResult<Self> {
        let (recording, recorder) = recorder.unzip();
        let streams = Arc::new(Streams::new(feed.buffer_seconds));
        let pump = runtime()?.spawn(streams::pump(streams.clone(), feed.events, feed.reconnects, feed.end));
        Ok(Self {
            streams,
            pump,
            state: Mutex::new(State::Open(Open {
                band: Arc::new(band),
                recorder,
            })),
            recording,
        })
    }

    fn read_chunk(&self, py: Python<'_>, name: &str, timeout: Option<f64>) -> PyResult<Option<Py<PyAny>>> {
        let index = stream_index(name)?;
        let limit = timeout.map(|t| seconds(t, "timeout")).transpose()?;
        let streams = self.streams.clone();
        let result = wait(py, async move {
            match limit {
                Some(limit) => tokio::time::timeout(limit, streams.read(index)).await.ok(),
                None => Some(streams.read(index).await),
            }
        })?;
        match result {
            None => Ok(None),
            Some(Ok(chunk)) => chunk_object(py, &chunk).map(Some),
            Some(Err(End::Finished)) => Err(Disconnected::new_err(format!(
                "no more {name} data: the connection has ended and everything was read"
            ))),
            Some(Err(End::Lost)) => Err(Disconnected::new_err(format!(
                "no more {name} data: the connection to the band was lost"
            ))),
        }
    }

    /// The live band, for calls that need one. The lock is released before returning, so a slow
    /// call never blocks other threads on it.
    fn live(&self) -> PyResult<Arc<frenz::Band>> {
        match &*self.state.lock().unwrap_or_else(PoisonError::into_inner) {
            State::Open(open) => Ok(open.band.clone()),
            State::Closed(_) => Err(FrenzError::new_err("the band is closed")),
        }
    }
}

#[pymethods]
impl Band {
    /// Name, firmware version and IMU calibration report.
    #[getter]
    fn info(&self) -> BandInfo {
        match &*self.state.lock().unwrap_or_else(PoisonError::into_inner) {
            State::Open(open) => open.band.info().into(),
            State::Closed(closed) => closed.info.clone(),
        }
    }

    /// Path of the recording file, or None when not recording.
    #[getter]
    fn recording(&self) -> Option<PathBuf> {
        self.recording.clone()
    }

    /// True until `close()` has been called.
    #[getter]
    fn is_open(&self) -> bool {
        matches!(
            &*self.state.lock().unwrap_or_else(PoisonError::into_inner),
            State::Open(_)
        )
    }

    /// Counters for the connection: notifications and decode errors per stream, sequence gaps,
    /// frames lost, reconnects, last battery and contact state, and whether the link is up.
    fn health<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let health = match &*self.state.lock().unwrap_or_else(PoisonError::into_inner) {
            State::Open(open) => open.band.health(),
            State::Closed(closed) => closed.health.clone(),
        };
        health_dict(py, &health, &self.streams)
    }

    /// Waits for new samples on one stream and returns them as a `Chunk`: everything that
    /// arrived since the last `read()` of that stream. Returns None if `timeout` seconds pass
    /// first. Raises `Disconnected` once the connection has ended and everything was read.
    #[pyo3(signature = (name, timeout=None))]
    fn read(&self, py: Python<'_>, name: &str, timeout: Option<f64>) -> PyResult<Option<Py<PyAny>>> {
        self.read_chunk(py, name, timeout)
    }

    /// Iterates over one stream: each item is the `Chunk` that `read()` would return. Ends when a
    /// replay finishes or the band is closed; raises `Disconnected` if the link is lost for good.
    fn stream(slf: Bound<'_, Self>, name: &str) -> PyResult<ChunkStream> {
        stream_index(name)?;
        Ok(ChunkStream {
            band: slf.unbind(),
            name: name.to_owned(),
        })
    }

    /// The most recent `seconds` of one stream as a `Chunk`, without consuming it: `read()` still
    /// returns these samples. Shorter than asked if less has been buffered.
    fn latest(&self, py: Python<'_>, name: &str, seconds: f64) -> PyResult<Py<PyAny>> {
        let index = stream_index(name)?;
        let span = crate::seconds(seconds, "seconds")?.as_secs_f64();
        let chunk = self
            .streams
            .latest(index, span)
            .ok_or_else(|| PyValueError::new_err(format!("unknown stream {name:?}")))?;
        chunk_object(py, &chunk)
    }

    /// Waits for the next event and returns it. Returns None if `timeout` seconds pass first
    /// (`timeout=0` only checks). Raises `Disconnected` once the connection has ended and every
    /// event was read.
    #[pyo3(signature = (timeout=None))]
    fn read_event(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<Option<BandEvent>> {
        let limit = timeout.map(|t| seconds(t, "timeout")).transpose()?;
        let streams = self.streams.clone();
        let result = wait(py, async move {
            match limit {
                Some(limit) => tokio::time::timeout(limit, streams.next_note()).await.ok(),
                None => Some(streams.next_note().await),
            }
        })?;
        match result {
            None => Ok(None),
            Some(Ok(note)) => BandEvent::new(py, &note).map(Some),
            Some(Err(_)) => Err(Disconnected::new_err(
                "no more events: the connection has ended and every event was read",
            )),
        }
    }

    /// Iterates over events as they happen. Ends when the connection is over. Meant for its own
    /// thread; in a loop that also reads data, use `read_event(timeout=0)`.
    fn events(slf: Bound<'_, Self>) -> EventStream {
        EventStream { band: slf.unbind() }
    }

    /// Sends a command without waiting for an answer, e.g. `band.send("set_prompt_volume", 0)`.
    /// See `openfrenz.COMMANDS` for names and arguments.
    #[pyo3(signature = (command, *args, **kwargs))]
    fn send(
        &self,
        py: Python<'_>,
        command: &str,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let command = parse_command(command, args, kwargs)?;
        let band = self.live()?;
        wait(py, band.send(command))?.map_err(to_py)
    }

    /// Sends a command and waits for the band's answer, e.g.
    /// `band.request("get_serial_number").value`. Raises `CommandTimeout` if none comes and
    /// `CommandFailed` if the band reports an error.
    #[pyo3(signature = (command, *args, **kwargs))]
    fn request(
        &self,
        py: Python<'_>,
        command: &str,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Response> {
        let command = parse_command(command, args, kwargs)?;
        let band = self.live()?;
        let response = wait(py, band.request(command))?.map_err(to_py)?;
        Response::new(py, &response)
    }

    /// Asks the band for its battery percentage.
    fn battery(&self, py: Python<'_>) -> PyResult<u8> {
        let band = self.live()?;
        wait(py, band.battery())?.map_err(to_py)
    }

    /// Writes a marker with the current time into the recording.
    fn mark(&self, label: &str) -> PyResult<()> {
        self.live()?.mark(label);
        Ok(())
    }

    /// Stops the streams, finishes the recording and disconnects. Returns the final `health()`.
    /// Calling it again returns the same counters.
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let open = {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            let closed = match &*state {
                State::Closed(closed) => return health_dict(py, &closed.health, &self.streams),
                State::Open(open) => Closed {
                    info: open.band.info().into(),
                    health: open.band.health(),
                },
            };
            match std::mem::replace(&mut *state, State::Closed(closed)) {
                State::Open(open) => open,
                State::Closed(_) => return Err(FrenzError::new_err("the band is closed")),
            }
        };
        // Readers waiting in other threads wake up and see the end.
        self.streams.end(End::Finished);
        self.pump.abort();
        // Another thread may still be inside a call on this band; it keeps the band alive until it
        // returns, and the band's tasks stop when the last reference goes.
        let stopped = Arc::try_unwrap(open.band).ok().map(|band| wait(py, band.stop()));
        // Finish the recording whatever happened above, so the file is always finalized.
        let finished = open.recorder.map(|recorder| wait(py, recorder.finish()));
        if let Some(health) = stopped.transpose()?.transpose().map_err(to_py)?
            && let State::Closed(closed) = &mut *self.state.lock().unwrap_or_else(PoisonError::into_inner)
        {
            closed.health = health;
        }
        if let Some(summary) = finished.transpose()? {
            summary.map_err(to_py)?;
        }
        self.health(py)
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }

    fn __repr__(&self) -> String {
        let state = if self.is_open() { "open" } else { "closed" };
        format!("Band(name={:?}, {state})", self.info().name)
    }
}

/// Iterator returned by `Band.stream()`.
#[pyclass(module = "openfrenz", frozen)]
struct ChunkStream {
    band: Py<Band>,
    name: String,
}

#[pymethods]
impl ChunkStream {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let band = self.band.get();
        match band.read_chunk(py, &self.name, None) {
            // A finished replay or a closed band is a normal end; a lost link is not.
            Err(error) if error.is_instance_of::<Disconnected>(py) && band.streams.ended() == Some(End::Finished) => {
                Ok(None)
            }
            other => other,
        }
    }
}

/// Iterator returned by `Band.events()`.
#[pyclass(module = "openfrenz", frozen)]
struct EventStream {
    band: Py<Band>,
}

#[pymethods]
impl EventStream {
    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<BandEvent>> {
        match self.band.get().read_event(py, None) {
            // The events themselves say how the connection ended, so the loop just stops.
            Err(error) if error.is_instance_of::<Disconnected>(py) => Ok(None),
            other => other,
        }
    }
}

fn export_error(error: frenz_export::Error) -> PyErr {
    let message = error.to_string();
    match error {
        frenz_export::Error::Sqlite(_) | frenz_export::Error::Io(_) | frenz_export::Error::Invalid(_) => {
            RecordingError::new_err(message)
        }
        _ => FrenzError::new_err(message),
    }
}

/// Exports a recording to `out_dir/<recording name>/` and returns that folder.
#[pyfunction]
fn export_recording(py: Python<'_>, recording: PathBuf, out_dir: PathBuf) -> PyResult<PathBuf> {
    py.detach(|| {
        let session = frenz_export::read(&recording)?;
        let name = recording
            .file_stem()
            .map_or_else(|| "session".into(), |stem| stem.to_string_lossy().to_string());
        let dir = out_dir.join(name);
        session.write(&dir)?;
        Ok(dir)
    })
    .map_err(export_error)
}

/// Counts what a recording holds, as `frenz summary` does: duration, samples and rate per stream,
/// and the counters a live connection would have ended with.
#[pyfunction]
fn summary(py: Python<'_>, recording: PathBuf) -> PyResult<Bound<'_, PyDict>> {
    let summary = py
        .detach(|| frenz_export::recording::summarize(&recording))
        .map_err(export_error)?;
    let (samples, rates) = (PyDict::new(py), PyDict::new(py));
    for (name, count) in &summary.samples {
        // The names Python uses for these streams everywhere else.
        let name = match *name {
            "accelerometer" => "accel",
            "gyroscope" => "gyro",
            other => other,
        };
        samples.set_item(name, count)?;
        rates.set_item(name, *count as f64 / summary.seconds)?;
    }
    let health = counters_dict(py, &summary.health)?;
    // A file is not a connection.
    health.del_item("connected")?;
    let dict = PyDict::new(py);
    dict.set_item("seconds", summary.seconds)?;
    dict.set_item("samples", samples)?;
    dict.set_item("rates_hz", rates)?;
    dict.set_item("health", health)?;
    Ok(dict)
}

/// Looks for bands nearby for `timeout` seconds.
#[pyfunction]
#[pyo3(signature = (timeout=8.0))]
fn scan(py: Python<'_>, timeout: f64) -> PyResult<Vec<Discovered>> {
    let found = wait(py, frenz::scan(seconds(timeout, "timeout")?))?.map_err(to_py)?;
    Ok(found
        .into_iter()
        .map(|d| Discovered {
            name: d.name,
            rssi: d.rssi,
        })
        .collect())
}

/// Connects to the band advertising `name`, starts its streams and (by default) a recording.
///
/// `record`: True writes `recordings/NAME-UNIXTIME.sqlite`; a file or directory path chooses the
/// location; False records nothing. `reconnect`: seconds to keep trying after the link is lost, or
/// None to never reconnect. `buffer_seconds`: how much of each stream is kept for `read()` and
/// `latest()`. `start=False` sends no start sequence.
// One Rust argument per Python keyword argument.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (name, *, record=None, reconnect=Some(300.0), scan_timeout=30.0, command_timeout=2.0, buffer_seconds=180.0, start=true))]
fn connect(
    py: Python<'_>,
    name: &str,
    record: Option<&Bound<'_, PyAny>>,
    reconnect: Option<f64>,
    scan_timeout: f64,
    command_timeout: f64,
    buffer_seconds: f64,
    start: bool,
) -> PyResult<Band> {
    let buffer_seconds = seconds(buffer_seconds, "buffer_seconds")?.as_secs_f64();
    let path = match record {
        Some(record) => recording_path(record, name)?,
        None => recording_path(PyBool::new(py, true).as_any(), name)?,
    };
    let options = ConnectOptions {
        scan_timeout: seconds(scan_timeout, "scan_timeout")?,
        command_timeout: seconds(command_timeout, "command_timeout")?,
        reconnect: reconnect.map(|r| seconds(r, "reconnect")).transpose()?,
        start: if start {
            StartSequence::Standard
        } else {
            StartSequence::None
        },
        ..ConnectOptions::default()
    };
    let reconnects = options.reconnect.is_some();
    let band = wait(py, frenz::Band::connect(name, options))?.map_err(to_py)?;
    // Subscribe before anything else can run, so the buffers see the first packet.
    let feed = Feed {
        events: band.subscribe(),
        buffer_seconds,
        reconnects,
        end: End::Lost,
    };
    let recorder = start_recorder(py, path, &band)?;
    Band::open(band, recorder, feed)
}

/// Plays a recording file back as if a band were sending it: no Bluetooth and no hardware.
///
/// `speed` is relative to the recording (1.0 keeps its pacing; 0 plays as fast as possible).
/// Commands such as `battery()` raise `Disconnected`, as there is no device. Nothing is recorded
/// unless `record` is given.
#[pyfunction]
#[pyo3(signature = (recording, *, speed=1.0, record=None, buffer_seconds=180.0))]
fn replay(
    py: Python<'_>,
    recording: PathBuf,
    speed: f64,
    record: Option<&Bound<'_, PyAny>>,
    buffer_seconds: f64,
) -> PyResult<Band> {
    let buffer_seconds = seconds(buffer_seconds, "buffer_seconds")?.as_secs_f64();
    let band = wait(py, frenz::Band::replay(&recording, ReplayOptions { speed }))?.map_err(to_py)?;
    let path = match record {
        Some(record) => recording_path(record, &band.info().name)?,
        None => None,
    };
    // Playback starts at the first subscription; the recorder still gets everything, because its
    // receiver was created with the band.
    let feed = Feed {
        events: band.subscribe(),
        buffer_seconds,
        reconnects: false,
        end: End::Finished,
    };
    let recorder = start_recorder(py, path, &band)?;
    Band::open(band, recorder, feed)
}

/// Writes a synthetic recording of `seconds` seconds to `path` (which must not exist), for trying
/// the tools without a band.
#[pyfunction]
#[pyo3(signature = (path, seconds=60))]
fn synth(path: PathBuf, seconds: u32) -> PyResult<()> {
    frenz::synth::write_recording(&path, seconds).map_err(to_py)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<Band>()?;
    module.add_class::<BandInfo>()?;
    module.add_class::<ChunkStream>()?;
    module.add_class::<EventStream>()?;
    module.add_class::<BandEvent>()?;
    module.add_class::<Response>()?;
    module.add_function(wrap_pyfunction!(export_recording, module)?)?;
    module.add_function(wrap_pyfunction!(summary, module)?)?;
    let commands = PyDict::new(py);
    for (name, parameters) in COMMANDS {
        commands.set_item(name, PyTuple::new(py, parameters)?)?;
    }
    module.add("COMMANDS", commands)?;
    module.add_class::<Discovered>()?;
    module.add_function(wrap_pyfunction!(scan, module)?)?;
    module.add_function(wrap_pyfunction!(connect, module)?)?;
    module.add_function(wrap_pyfunction!(replay, module)?)?;
    module.add_function(wrap_pyfunction!(synth, module)?)?;
    module.add("FrenzError", py.get_type::<FrenzError>())?;
    module.add("BandNotFound", py.get_type::<BandNotFound>())?;
    module.add("BluetoothError", py.get_type::<BluetoothError>())?;
    module.add("BluetoothUnavailable", py.get_type::<BluetoothUnavailable>())?;
    module.add("CommandTimeout", py.get_type::<CommandTimeout>())?;
    module.add("CommandFailed", py.get_type::<CommandFailed>())?;
    module.add("Disconnected", py.get_type::<Disconnected>())?;
    module.add("RecordingError", py.get_type::<RecordingError>())?;
    Ok(())
}
