//! Discovering, connecting to, and commanding a band.

use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use btleplug::api::ValueNotification;
use btleplug::api::{
    Central, CentralEvent, CharPropFlags, Characteristic as BleCharacteristic, Manager as _, Peripheral as _,
    ScanFilter, WriteType,
};
use btleplug::platform::{Adapter, Manager, Peripheral, PeripheralId};
use frenz_proto::{Characteristic, Command, Packet, Response, ResponseBody};
use futures::{Stream, StreamExt};
use tokio::sync::{Notify, RwLock, broadcast, mpsc};
use tokio::task::JoinHandle;

use crate::clock::{mono_ns, wall_ns};
use crate::error::Error;
use crate::event::{Decoder, Event, Health, RawRecord};

/// Capacity of the event and raw-record broadcast channels (≈ 3 minutes of all streams).
const CHANNEL_CAPACITY: usize = 1 << 14;

/// A band seen while scanning.
#[derive(Debug, Clone)]
pub struct Discovered {
    /// Advertised name.
    pub name: String,
    /// Signal strength at discovery.
    pub rssi: Option<i16>,
}

/// What to send after subscribing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StartSequence {
    /// The standard start sequence (verified on hardware): firmware query, time sync,
    /// prompt volume 0, periodic heart rate, on-connected, stream start, IMU calibration query.
    #[default]
    Standard,
    /// Send nothing.
    None,
}

/// Connection options.
#[derive(Debug, Clone)]
pub struct ConnectOptions {
    /// How long to scan for the named band.
    pub scan_timeout: Duration,
    /// Commands sent after subscribing.
    pub start: StartSequence,
    /// How long to wait for a command response.
    pub command_timeout: Duration,
    /// After the link is lost, keep trying to reconnect for this long (`None`: do not reconnect).
    /// The band powers itself off about 33 s after being taken off, so a reconnect only succeeds
    /// if it is still on (or is switched on again) within this time.
    pub reconnect: Option<Duration>,
    /// Characteristics to subscribe to. The default is every notify characteristic of the band.
    pub notify: Vec<Characteristic>,
}

impl Default for ConnectOptions {
    fn default() -> Self {
        Self {
            scan_timeout: Duration::from_secs(30),
            start: StartSequence::default(),
            command_timeout: Duration::from_secs(2),
            reconnect: Some(Duration::from_secs(300)),
            notify: Characteristic::NOTIFY.to_vec(),
        }
    }
}

/// Facts about the connected band.
#[derive(Debug, Clone, Default)]
pub struct BandInfo {
    /// Advertised name.
    pub name: String,
    /// Firmware version string (from the start sequence).
    pub firmware: Option<String>,
    /// IMU calibration report `(std, diff)` (from the start sequence).
    pub imu_calibration: Option<([u8; 3], [u16; 3])>,
    /// GATT table at connect: `(service, characteristic, properties)`.
    pub gatt: Vec<(String, String, String)>,
}

async fn first_adapter() -> Result<Adapter, Error> {
    Manager::new()
        .await?
        .adapters()
        .await?
        .into_iter()
        .next()
        .ok_or(Error::NoAdapter)
}

async fn frenz_peripherals(adapter: &Adapter) -> Result<Vec<(Discovered, Peripheral)>, Error> {
    let mut found = Vec::new();
    for p in adapter.peripherals().await? {
        if let Some(props) = p.properties().await?
            && let Some(name) = props.local_name
            && name.to_ascii_uppercase().starts_with("FRENZ")
        {
            found.push((Discovered { name, rssi: props.rssi }, p));
        }
    }
    Ok(found)
}

/// Scans for bands for `duration`, strongest signal first.
pub async fn scan(duration: Duration) -> Result<Vec<Discovered>, Error> {
    let adapter = first_adapter().await?;
    adapter.start_scan(ScanFilter::default()).await?;
    tokio::time::sleep(duration).await;
    let mut found: Vec<_> = frenz_peripherals(&adapter).await?.into_iter().map(|(d, _)| d).collect();
    adapter.stop_scan().await.ok();
    found.sort_by_key(|d| std::cmp::Reverse(d.rssi.unwrap_or(i16::MIN)));
    Ok(found)
}

/// With no notification at all for this long, the link is checked and treated as lost if down.
const STALL: Duration = Duration::from_secs(10);

type Notifications = Pin<Box<dyn Stream<Item = ValueNotification> + Send>>;
type CentralEvents = Pin<Box<dyn Stream<Item = CentralEvent> + Send>>;

struct Link {
    peripheral: Peripheral,
    characteristics: HashMap<Characteristic, BleCharacteristic>,
}

enum ToDecoder {
    Notification(i64, Characteristic, Arc<[u8]>),
    /// The link was re-established: sequence tracking starts afresh.
    Reset,
    /// A replay reached a recorded command that started the streams, at this (replay) time.
    StreamStart(i64),
    /// A replay reached its end. Sent through the decoder so the disconnect is reported after
    /// every notification queued before it.
    ReplayEnded,
}

/// State shared by the handle and its background tasks.
struct Shared {
    name: String,
    options: ConnectOptions,
    events: broadcast::Sender<Event>,
    raw: broadcast::Sender<RawRecord>,
    health: Mutex<Health>,
    last_start_ns: AtomicI64,
    info: Mutex<BandInfo>,
    link: RwLock<Option<Link>>,
    closing: AtomicBool,
    /// Identity of the device first connected to; reconnection accepts only this device.
    device: Mutex<Option<PeripheralId>>,
    /// Signalled by the first subscription; a replay waits for it so no event is missed.
    subscribed: Notify,
}

impl Shared {
    fn new(name: &str, options: ConnectOptions) -> (Arc<Self>, broadcast::Receiver<RawRecord>) {
        let (events, _) = broadcast::channel(CHANNEL_CAPACITY);
        let (raw, initial_raw) = broadcast::channel(CHANNEL_CAPACITY);
        let shared = Arc::new(Self {
            name: name.to_owned(),
            options,
            events,
            raw,
            health: Mutex::new(Health::default()),
            last_start_ns: AtomicI64::new(i64::MIN / 2),
            info: Mutex::new(BandInfo {
                name: name.to_owned(),
                ..BandInfo::default()
            }),
            link: RwLock::new(None),
            closing: AtomicBool::new(false),
            device: Mutex::new(None),
            subscribed: Notify::new(),
        });
        (shared, initial_raw)
    }

    fn mark(&self, label: &str) {
        let _ = self.raw.send(RawRecord::Marker {
            mono_ns: mono_ns(),
            wall_ns: wall_ns(),
            label: label.to_owned(),
        });
    }

    fn with_health<R>(&self, f: impl FnOnce(&mut Health) -> R) -> R {
        f(&mut self.health.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// Scans for the band, connects, and subscribes to its streams.
    async fn establish(&self, scan_timeout: Duration) -> Result<(Notifications, CentralEvents, PeripheralId), Error> {
        let adapter = first_adapter().await?;
        adapter.start_scan(ScanFilter::default()).await?;
        let deadline = tokio::time::Instant::now() + scan_timeout;
        // After the first connection, only the same physical device is accepted: a name can be
        // advertised by anything.
        let pinned = self.device.lock().unwrap_or_else(PoisonError::into_inner).clone();
        let peripheral = loop {
            if let Some((_, p)) = frenz_peripherals(&adapter)
                .await?
                .into_iter()
                .find(|(d, p)| d.name == self.name && pinned.as_ref().is_none_or(|id| *id == p.id()))
            {
                break p;
            }
            if tokio::time::Instant::now() >= deadline {
                adapter.stop_scan().await.ok();
                return Err(Error::NotFound {
                    name: self.name.clone(),
                    timeout: scan_timeout,
                });
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        };
        adapter.stop_scan().await.ok();
        let connect_timeout = Duration::from_secs(15);
        tokio::time::timeout(connect_timeout, peripheral.connect())
            .await
            .map_err(|_| Error::NotFound {
                name: self.name.clone(),
                timeout: connect_timeout,
            })??;
        peripheral.discover_services().await?;

        let mut characteristics = HashMap::new();
        let mut gatt = Vec::new();
        for c in peripheral.characteristics() {
            gatt.push((
                c.service_uuid.to_string(),
                c.uuid.to_string(),
                format!("{:?}", c.properties),
            ));
            if let Some(known) = Characteristic::from_uuid(c.uuid.as_u128()) {
                characteristics.insert(known, c);
            }
        }
        for required in Characteristic::NOTIFY.into_iter().chain([Characteristic::CommandRx]) {
            if !characteristics.contains_key(&required) {
                return Err(Error::MissingCharacteristic(required.name()));
            }
        }
        // Obtain the notification stream before subscribing so no early notification is missed.
        let notifications = peripheral.notifications().await?;
        let central_events = adapter.events().await?;
        let id = peripheral.id();
        self.device
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert_with(|| id.clone());
        self.info.lock().unwrap_or_else(PoisonError::into_inner).gatt = gatt;
        self.mark("connected");
        // Subscribe only to the band's own notify characteristics; other services drop the link.
        for c in &self.options.notify {
            if let Some(ble) = characteristics.get(c)
                && ble.properties.contains(CharPropFlags::NOTIFY)
            {
                peripheral.subscribe(ble).await?;
            }
        }
        self.mark("subscribed_all");
        *self.link.write().await = Some(Link {
            peripheral,
            characteristics,
        });
        self.with_health(|h| h.connected = true);
        Ok((notifications, central_events, id))
    }

    async fn send(&self, command: Command) -> Result<(), Error> {
        let link = self.link.read().await;
        let link = link.as_ref().ok_or(Error::Disconnected)?;
        let rx = link
            .characteristics
            .get(&Characteristic::CommandRx)
            .ok_or(Error::MissingCharacteristic("command_rx"))?;
        let encoded = command.encode();
        if matches!(command, Command::StreamControl { enable: true, .. }) {
            self.last_start_ns.store(mono_ns(), Ordering::Relaxed);
        }
        let result = link
            .peripheral
            .write(rx, encoded.as_bytes(), WriteType::WithoutResponse)
            .await;
        let _ = self.raw.send(RawRecord::Write {
            mono_ns: mono_ns(),
            wall_ns: wall_ns(),
            label: format!("{command:?}"),
            bytes: encoded.as_bytes().into(),
            error: result.as_ref().err().map(ToString::to_string),
        });
        Ok(result?)
    }

    async fn request(&self, command: Command) -> Result<Response, Error> {
        let mut events = self.events.subscribe();
        self.send(command).await?;
        let opcode = command.opcode();
        let timeout = self.options.command_timeout;
        let response = tokio::time::timeout(timeout, async {
            loop {
                match events.recv().await {
                    Ok(Event::Packet { packet, .. }) => {
                        if let Packet::Response(r) = packet.as_ref()
                            && r.opcode == opcode
                        {
                            return Ok(r.clone());
                        }
                    }
                    Ok(Event::Disconnected { .. }) | Err(broadcast::error::RecvError::Closed) => {
                        return Err(Error::Disconnected);
                    }
                    Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                }
            }
        })
        .await
        .map_err(|_| Error::Timeout { opcode, timeout })??;
        if response.ok() {
            Ok(response)
        } else {
            Err(Error::CommandFailed {
                opcode,
                status: response.status,
            })
        }
    }

    async fn run_start_sequence(&self) -> Result<(), Error> {
        let now = u32::try_from(wall_ns() / 1_000_000_000).unwrap_or(u32::MAX);
        let pause = Duration::from_millis(100);
        match self.options.start {
            StartSequence::None => return Ok(()),
            StartSequence::Standard => {
                if let Ok(r) = self.request(Command::GetFirmwareVersion).await
                    && let Ok(ResponseBody::FirmwareVersion(v)) = r.body()
                {
                    self.info.lock().unwrap_or_else(PoisonError::into_inner).firmware = Some(v.to_owned());
                }
                self.send(Command::TimeSync(now)).await?;
                self.send(Command::SetPromptVolume(0)).await?;
                tokio::time::sleep(pause).await;
                self.send(Command::HeartRatePeriodic(true)).await?;
                tokio::time::sleep(pause).await;
                self.send(Command::OnConnected).await?;
                self.send(Command::StreamControl {
                    mask: 0xff,
                    enable: true,
                })
                .await?;
                tokio::time::sleep(pause).await;
            }
        }
        if let Ok(r) = self.request(Command::GetImuCalibration).await
            && let Ok(ResponseBody::ImuCalibration { std, diff }) = r.body()
        {
            self.info.lock().unwrap_or_else(PoisonError::into_inner).imu_calibration = Some((std, diff));
        }
        // Without this the band powers itself off 10 minutes after the stream start.
        self.send(Command::StartSession).await?;
        self.mark("start_sequence_sent");
        Ok(())
    }
}

/// Forwards notifications until the link is lost; then, if enabled, reconnects and carries on.
async fn supervise(
    shared: Arc<Shared>,
    mut notifications: Notifications,
    mut central_events: CentralEvents,
    mut id: PeripheralId,
    to_decoder: mpsc::UnboundedSender<ToDecoder>,
) {
    loop {
        // Arrival path: timestamp, copy, hand off. No parsing, no blocking I/O.
        loop {
            tokio::select! {
                n = notifications.next() => {
                    let Some(n) = n else { break };
                    let (mono, wall) = (mono_ns(), wall_ns());
                    let Some(characteristic) = Characteristic::from_uuid(n.uuid.as_u128()) else { continue };
                    let bytes: Arc<[u8]> = n.value.into();
                    let _ = shared.raw.send(RawRecord::Notification { mono_ns: mono, wall_ns: wall, characteristic, bytes: bytes.clone() });
                    if to_decoder.send(ToDecoder::Notification(mono, characteristic, bytes)).is_err() {
                        return;
                    }
                }
                e = central_events.next() => match e {
                    Some(CentralEvent::DeviceDisconnected(d)) if d == id => break,
                    Some(_) => {}
                    None => break,
                },
                () = tokio::time::sleep(STALL) => {
                    let link = shared.link.read().await;
                    let alive = match link.as_ref() {
                        Some(l) => l.peripheral.is_connected().await.unwrap_or(false),
                        None => false,
                    };
                    if !alive { break }
                }
            }
        }
        if let Some(link) = shared.link.write().await.take() {
            link.peripheral.disconnect().await.ok();
        }
        shared.with_health(|h| h.connected = false);
        shared.mark("disconnected");
        let _ = shared.events.send(Event::Disconnected { mono_ns: mono_ns() });
        let Some(window) = shared
            .options
            .reconnect
            .filter(|_| !shared.closing.load(Ordering::Relaxed))
        else {
            return;
        };

        let deadline = tokio::time::Instant::now() + window;
        let mut attempt = 0u32;
        loop {
            if shared.closing.load(Ordering::Relaxed) {
                return;
            }
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                shared.mark("reconnect_gave_up");
                let _ = shared.events.send(Event::ReconnectFailed { mono_ns: mono_ns() });
                return;
            }
            attempt += 1;
            let _ = shared.events.send(Event::Reconnecting {
                mono_ns: mono_ns(),
                attempt,
            });
            match shared.establish(remaining.min(Duration::from_secs(10))).await {
                Ok((n, c, i)) => {
                    notifications = n;
                    central_events = c;
                    id = i;
                    break;
                }
                Err(_) => {
                    let backoff = Duration::from_secs(u64::from(attempt.min(5)));
                    tokio::time::sleep(backoff.min(deadline.saturating_duration_since(tokio::time::Instant::now())))
                        .await;
                }
            }
        }
        let _ = to_decoder.send(ToDecoder::Reset);
        shared.with_health(|h| h.reconnects += 1);
        shared.mark("reconnected");
        // The start sequence waits for responses, which this task delivers, so run it alongside.
        let again = shared.clone();
        tokio::spawn(async move {
            again.run_start_sequence().await.ok();
            let _ = again.events.send(Event::Reconnected { mono_ns: mono_ns() });
        });
    }
}

/// Starts the task that decodes notifications into events and keeps the health counters.
fn spawn_decoder(shared: Arc<Shared>) -> (mpsc::UnboundedSender<ToDecoder>, JoinHandle<()>) {
    let (to_decoder, mut decoder_rx) = mpsc::unbounded_channel::<ToDecoder>();
    let task = tokio::spawn(async move {
        let mut decoder = Decoder::default();
        let mut out = Vec::new();
        while let Some(message) = decoder_rx.recv().await {
            match message {
                ToDecoder::Reset => decoder.reset(),
                ToDecoder::StreamStart(mono) => shared.last_start_ns.store(mono, Ordering::Relaxed),
                ToDecoder::ReplayEnded => {
                    shared.with_health(|h| h.connected = false);
                    shared.mark("disconnected");
                    let _ = shared.events.send(Event::Disconnected { mono_ns: mono_ns() });
                }
                ToDecoder::Notification(mono, characteristic, bytes) => {
                    let last_start = shared.last_start_ns.load(Ordering::Relaxed);
                    shared.with_health(|h| decoder.handle(mono, characteristic, &bytes, last_start, h, &mut out));
                    for event in out.drain(..) {
                        let _ = shared.events.send(event);
                    }
                }
            }
        }
    });
    (to_decoder, task)
}

/// Options for [`Band::replay`].
#[derive(Debug, Clone, Copy)]
pub struct ReplayOptions {
    /// Playback speed relative to the recording: 1.0 delivers notifications with their recorded
    /// spacing, 10.0 ten times faster. Zero, negative or non-finite values deliver them as fast as
    /// possible.
    pub speed: f64,
}

impl Default for ReplayOptions {
    fn default() -> Self {
        Self { speed: 1.0 }
    }
}

/// What a recording file holds that a replay needs.
struct Replayable {
    name: String,
    firmware: Option<String>,
    /// `(recorded monotonic time, what happened)` in recorded order.
    records: Vec<(i64, Recorded)>,
}

/// One thing a recording holds that a replay reproduces.
enum Recorded {
    /// A notification from the band.
    Notification(Characteristic, Arc<[u8]>),
    /// A command that was written to the band. Nothing is written during a replay, but the
    /// decoder needs to know when streams were started, and a recording made of a replay should
    /// hold the same commands as the original.
    Write {
        label: String,
        bytes: Arc<[u8]>,
        error: Option<String>,
    },
}

fn load_replayable(path: &Path) -> Result<Replayable, Error> {
    let db = rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let meta = |key: &str| -> Result<Option<String>, Error> {
        let mut statement = db.prepare("SELECT value FROM meta WHERE key = ?1")?;
        let mut rows = statement.query([key])?;
        Ok(match rows.next()? {
            Some(row) => Some(row.get(0)?),
            None => None,
        })
    };
    let name = meta("device_name")?.unwrap_or_else(|| "FRENZ-REPLAY".to_owned());
    let firmware = meta("firmware")?.filter(|f| !f.is_empty());
    let mut statement = db.prepare("SELECT mono_ns, char, payload FROM notifications ORDER BY id")?;
    let mut rows = statement.query([])?;
    let mut records = Vec::new();
    while let Some(row) = rows.next()? {
        let (mono, uuid, payload): (i64, String, Vec<u8>) = (row.get(0)?, row.get(1)?, row.get(2)?);
        // Rows from characteristics this client does not know are skipped, as they are live.
        if let Some(characteristic) = uuid::Uuid::parse_str(&uuid)
            .ok()
            .and_then(|u| Characteristic::from_uuid(u.as_u128()))
        {
            records.push((mono, Recorded::Notification(characteristic, payload.into())));
        }
    }
    let mut statement = db.prepare("SELECT mono_ns, label, payload, error FROM writes ORDER BY id")?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let (mono, label, payload, error): (i64, String, Vec<u8>, Option<String>) =
            (row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?);
        records.push((
            mono,
            Recorded::Write {
                label,
                bytes: payload.into(),
                error,
            },
        ));
    }
    // Recorded order across both tables. The sort is stable, so each table keeps its own order,
    // and a command comes before a notification with the same timestamp.
    records.sort_by_key(|(mono, record)| (*mono, matches!(record, Recorded::Notification(..))));
    Ok(Replayable {
        name,
        firmware,
        records,
    })
}

/// True for the bytes of a command that turns streams on.
fn starts_streams(bytes: &[u8]) -> bool {
    let start = Command::StreamControl { mask: 0, enable: true }.encode();
    // Opcode, stream mask, enable flag: any mask counts, as it does for a live write.
    matches!((bytes, start.as_bytes()), ([opcode, _, enable], [start_opcode, _, start_enable])
        if opcode == start_opcode && enable == start_enable)
}

/// Delivers recorded notifications on the path live ones take, then reports a disconnect.
async fn replay(
    shared: Arc<Shared>,
    records: Vec<(i64, Recorded)>,
    speed: f64,
    to_decoder: mpsc::UnboundedSender<ToDecoder>,
) {
    shared.subscribed.notified().await;
    let paced = speed.is_finite() && speed > 0.0;
    let started = tokio::time::Instant::now();
    let first = records
        .iter()
        .find(|(_, record)| matches!(record, Recorded::Notification(..)))
        .map_or(0, |(mono, _)| *mono);
    for (recorded, record) in records {
        if shared.closing.load(Ordering::Relaxed) {
            return;
        }
        if paced {
            let offset = Duration::from_secs_f64((recorded - first).max(0) as f64 / 1e9 / speed);
            tokio::time::sleep_until(started + offset).await;
        } else {
            tokio::task::yield_now().await;
        }
        // Arrival is stamped now, as for a live notification.
        let (mono, wall) = (mono_ns(), wall_ns());
        let (characteristic, bytes) = match record {
            Recorded::Notification(characteristic, bytes) => (characteristic, bytes),
            Recorded::Write { label, bytes, error } => {
                // The decoder learns of a stream start in order with the notifications around it.
                if starts_streams(&bytes) && to_decoder.send(ToDecoder::StreamStart(mono)).is_err() {
                    return;
                }
                let _ = shared.raw.send(RawRecord::Write {
                    mono_ns: mono,
                    wall_ns: wall,
                    label,
                    bytes,
                    error,
                });
                continue;
            }
        };
        let _ = shared.raw.send(RawRecord::Notification {
            mono_ns: mono,
            wall_ns: wall,
            characteristic,
            bytes: bytes.clone(),
        });
        if to_decoder
            .send(ToDecoder::Notification(mono, characteristic, bytes))
            .is_err()
        {
            return;
        }
    }
    let _ = to_decoder.send(ToDecoder::ReplayEnded);
}

/// A connected band.
///
/// If the link drops, the band is reconnected automatically (see [`ConnectOptions::reconnect`]);
/// subscribers, recorders and this handle keep working across the interruption. Dropping a `Band`
/// stops its background tasks but does not send the stop sequence; call [`Band::stop`] for an
/// orderly end.
pub struct Band {
    shared: Arc<Shared>,
    tasks: Vec<JoinHandle<()>>,
    /// Receiver created before subscribing, so the first recorder sees everything since connect.
    initial_raw: Mutex<Option<broadcast::Receiver<RawRecord>>>,
}

impl Band {
    /// Scans for a band advertising `name`, connects, subscribes to all of its streams and runs the
    /// configured start sequence.
    pub async fn connect(name: &str, options: ConnectOptions) -> Result<Self, Error> {
        let scan_timeout = options.scan_timeout;
        let (shared, initial_raw) = Shared::new(name, options);
        let (notifications, central_events, id) = shared.establish(scan_timeout).await?;

        let (to_decoder, decoder) = spawn_decoder(shared.clone());
        let mut tasks = vec![decoder];
        tasks.push(tokio::spawn(supervise(
            shared.clone(),
            notifications,
            central_events,
            id,
            to_decoder,
        )));
        let band = Self {
            shared,
            tasks,
            initial_raw: Mutex::new(Some(initial_raw)),
        };
        band.shared.run_start_sequence().await?;
        Ok(band)
    }

    /// Plays a recording file back as if a band were sending it: no Bluetooth and no hardware.
    ///
    /// The recorded notifications go through the same decoder, events, health counters and raw
    /// record stream as live ones, stamped with the time they are delivered. When the recording
    /// ends, [`Event::Disconnected`] is sent. Playback starts at the first call to
    /// [`Band::subscribe`], [`Band::subscribe_raw`] or [`Band::subscribe_raw_from_start`], so that
    /// subscriber sees everything from the first notification. There is no device to command, so [`Band::send`] and
    /// [`Band::request`] fail with [`Error::Disconnected`]. Meant for tests, demos and developing
    /// against the client without a band.
    pub async fn replay(recording: impl AsRef<Path>, options: ReplayOptions) -> Result<Self, Error> {
        let path = recording.as_ref().to_path_buf();
        let loaded = tokio::task::spawn_blocking(move || load_replayable(&path))
            .await
            .map_err(std::io::Error::other)??;
        let (shared, initial_raw) = Shared::new(
            &loaded.name,
            ConnectOptions {
                reconnect: None,
                start: StartSequence::None,
                ..ConnectOptions::default()
            },
        );
        shared.info.lock().unwrap_or_else(PoisonError::into_inner).firmware = loaded.firmware;
        shared.with_health(|h| h.connected = true);
        // Treat the start of playback as a stream start, as the recorded session began with one.
        shared.last_start_ns.store(mono_ns(), Ordering::Relaxed);
        shared.mark("connected");
        let (to_decoder, decoder) = spawn_decoder(shared.clone());
        let player = tokio::spawn(replay(shared.clone(), loaded.records, options.speed, to_decoder));
        Ok(Self {
            shared,
            tasks: vec![decoder, player],
            initial_raw: Mutex::new(Some(initial_raw)),
        })
    }

    /// Facts about the band gathered at connect.
    pub fn info(&self) -> BandInfo {
        self.shared.info.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Snapshot of the connection's counters.
    pub fn health(&self) -> Health {
        self.shared.with_health(|h| h.clone())
    }

    /// Subscribes to decoded events. A subscriber that falls more than ~16k events behind receives
    /// `RecvError::Lagged` with the number of events it missed.
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        let receiver = self.shared.events.subscribe();
        self.shared.subscribed.notify_one();
        receiver
    }

    /// Subscribes to raw records (notifications, writes, markers) in arrival order, from now on.
    pub fn subscribe_raw(&self) -> broadcast::Receiver<RawRecord> {
        let receiver = self.shared.raw.subscribe();
        self.shared.subscribed.notify_one();
        receiver
    }

    /// Like [`Band::subscribe_raw`], but the first call receives every record since the connection
    /// was set up (the connect markers, start sequence and first notifications). Later calls
    /// subscribe from now on. Recorders use this so a recording is complete.
    pub fn subscribe_raw_from_start(&self) -> broadcast::Receiver<RawRecord> {
        let receiver = self
            .initial_raw
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .unwrap_or_else(|| self.shared.raw.subscribe());
        self.shared.subscribed.notify_one();
        receiver
    }

    /// Adds a marker to the raw record stream.
    pub fn mark(&self, label: &str) {
        self.shared.mark(label);
    }

    /// Writes a command without waiting for a response. Fails with [`Error::Disconnected`] while
    /// the link is down.
    pub async fn send(&self, command: Command) -> Result<(), Error> {
        self.shared.send(command).await
    }

    /// Writes a command and waits for the response with the same opcode.
    pub async fn request(&self, command: Command) -> Result<Response, Error> {
        self.shared.request(command).await
    }

    /// Queries the battery percentage.
    pub async fn battery(&self) -> Result<u8, Error> {
        match self.request(Command::GetBattery).await?.body() {
            Ok(ResponseBody::Battery(p)) => Ok(p),
            _ => Err(Error::CommandFailed {
                opcode: 0x02,
                status: 0xff,
            }),
        }
    }

    /// Stops streaming, sends the session-end notice, and disconnects. No reconnection follows.
    pub async fn stop(mut self) -> Result<Health, Error> {
        self.shared.closing.store(true, Ordering::Relaxed);
        if self.health().connected {
            self.send(Command::StreamControl {
                mask: 0xff,
                enable: false,
            })
            .await
            .ok();
            tokio::time::sleep(Duration::from_millis(100)).await;
            self.send(Command::EndSession).await.ok();
            tokio::time::sleep(Duration::from_millis(500)).await;
            self.mark("stop_sequence_sent");
            if let Some(link) = self.shared.link.write().await.take() {
                link.peripheral.disconnect().await.ok();
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        for task in self.tasks.drain(..) {
            task.abort();
        }
        Ok(self.health())
    }
}

impl Drop for Band {
    fn drop(&mut self) {
        self.shared.closing.store(true, Ordering::Relaxed);
        for task in &self.tasks {
            task.abort();
        }
    }
}
