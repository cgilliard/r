//! Command-line argument parsing.

use std::ffi::OsString;
use std::fmt;
use std::net::SocketAddr;
use std::path::PathBuf;

pub const USAGE: &str = "\
Usage: r [OPTIONS]

Runs a node on the chain in <PATH>/chain (creating it if needed): it finds
peers (from --peer, the peer list saved with the chain, and what peers pass
on; with none of those, from the seed nodes tabernacle1.root.sx:3737 and
tabernacle2.root.sx:3737), syncs the chain, verifies and follows peers' blocks, relays
transactions, and mines (unless --no-mine) once caught up. Its log goes to
<PATH>/node.log; the terminal takes wallet commands (type help): payments go
through files, with send, receive and finalize.

Options:
      --data-dir <PATH>      Directory for the database [default: ~/.tabernacle]
      --listen <ADDR>        Accept peers on this UDP address
                             [default: 0.0.0.0:3737]
      --no-listen            Don't accept peers: only dial out (such as a
                             second node on the same machine)
      --peer <ADDR>          Connect to a peer (may be repeated)
      --sync-depth <BLOCKS>  A new node downloads the state at BLOCKS below
                             the best peer's tip, then the blocks from there,
                             instead of every block [default: 100]; peers
                             must keep that much (see --snap-sync-limit)
      --snap-sync-limit <BLOCKS>
                             Keep the state tree's history for the last
                             BLOCKS blocks, to serve fast-syncing nodes the
                             state at any of them; also the deepest reorg
                             followed [default: 200]
      --keep-blocks <BLOCKS> Keep the last BLOCKS blocks (at least
                             --snap-sync-limit), for peers catching up;
                             older ones are deleted. 0 keeps every block
                             [default: 1000]
      --compact-db           Rewrite the database compactly before starting
                             (takes a while on a large one): reclaims the
                             space freed by pruning and spent outputs
      --max-peers <NUM>      Most peers connected at once, inbound and
                             outbound [default: 64]
      --no-mine              Don't mine (saves CPU): only follow and serve
                             the chain, verifying every block, and run the
                             wallet
      --coin-maturity <BLOCKS>
                             Confirmations before this node's mining rewards
                             can be spent [default: 6]
      --restore              Restore the wallet from its 24 recovery words
                             (asked for at the start; needs a new --data-dir)
      --debug-no-proof-validation
                             Testing only: make and check no chain proofs
                             (blocks carry an empty one), so large blocks
                             can be built quickly; every other check stays.
                             Such nodes form their own network (they can't
                             reach nodes without it), and the data directory
                             is marked for it. Enables the debug-fill command
      --status               Print the stored chain's status and exit
  -h, --help                 Print help
  -V, --version              Print version";

/// `~/.tabernacle` (`.tabernacle` in the current directory if `HOME` isn't
/// set).
pub fn default_data_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map_or_else(PathBuf::new, PathBuf::from)
        .join(".tabernacle")
}
/// Where a node accepts peers unless told otherwise: every interface, on
/// the network's port (the seeds').
pub const DEFAULT_LISTEN: SocketAddr = SocketAddr::V4(std::net::SocketAddrV4::new(
    std::net::Ipv4Addr::UNSPECIFIED,
    3737,
));
pub const DEFAULT_SYNC_DEPTH: u64 = 100;
pub const DEFAULT_MAX_PEERS: usize = 64;
pub const DEFAULT_COIN_MATURITY: u64 = crate::cli::DEFAULT_MATURITY;
pub const DEFAULT_SNAP_SYNC_LIMIT: u64 = crate::ledger::DEFAULT_SNAP_SYNC_LIMIT;
pub const DEFAULT_KEEP_BLOCKS: u64 = crate::ledger::DEFAULT_KEEP_BLOCKS;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Args {
    pub data_dir: PathBuf,
    /// Only follow and serve the chain, without mining.
    pub no_mine: bool,
    /// Print the stored chain's status and exit.
    pub status: bool,
    /// The address to accept peers on.
    pub listen: Option<SocketAddr>,
    /// Peers to connect to.
    pub peers: Vec<SocketAddr>,
    /// Fast sync's distance below the best tip.
    pub sync_depth: u64,
    /// Most open peer sessions.
    pub max_peers: usize,
    /// Blocks of state history kept for fast-syncing peers.
    pub snap_sync_limit: u64,
    /// Blocks whose bodies are kept (0: all).
    pub keep_blocks: u64,
    /// Compact the database before starting.
    pub compact_db: bool,
    /// Confirmations before our mining rewards can be spent.
    pub coin_maturity: u64,
    /// Restore the wallet from recovery words.
    pub restore: bool,
    /// Testing: make and check no chain proofs.
    pub debug_no_proofs: bool,
}

/// What the command line asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Run(Args),
    Help,
    Version,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgsError {
    /// An option that takes a value was given none.
    MissingValue(&'static str),
    /// An option was given more than once.
    Duplicate(&'static str),
    /// An option that takes no value was given one (`--help=x`).
    UnexpectedValue(&'static str),
    /// An option's value doesn't parse.
    InvalidValue(&'static str),
    /// Two options that can't go together.
    Conflict(&'static str, &'static str),
    Unknown(OsString),
}

impl fmt::Display for ArgsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingValue(opt) => write!(f, "option '{opt}' requires a value"),
            Self::Duplicate(opt) => write!(f, "option '{opt}' was given more than once"),
            Self::UnexpectedValue(opt) => write!(f, "option '{opt}' does not take a value"),
            Self::InvalidValue(opt) => write!(f, "invalid value for option '{opt}'"),
            Self::Conflict(a, b) => write!(f, "options '{a}' and '{b}' can't be used together"),
            Self::Unknown(arg) => write!(f, "unexpected argument '{}'", arg.to_string_lossy()),
        }
    }
}

impl std::error::Error for ArgsError {}

impl Args {
    /// Parses the process's arguments, skipping the program name.
    pub fn from_env() -> Result<Command, ArgsError> {
        Self::parse(std::env::args_os().skip(1))
    }

    /// Parses `args`, which must not include the program name.
    ///
    /// Values may be given as `--opt value` or `--opt=value`. The `=` form
    /// requires the value to be valid UTF-8; use the space form otherwise.
    /// `--help` and `--version` take effect as soon as they are seen.
    pub fn parse<I>(args: I) -> Result<Command, ArgsError>
    where
        I: IntoIterator,
        I::Item: Into<OsString>,
    {
        let mut data_dir: Option<PathBuf> = None;
        let mut no_mine = false;
        let mut status = false;
        let mut listen: Option<SocketAddr> = None;
        let mut no_listen = false;
        let mut peers: Vec<SocketAddr> = Vec::new();
        let mut sync_depth: Option<u64> = None;
        let mut max_peers: Option<usize> = None;
        let mut snap_sync_limit: Option<u64> = None;
        let mut keep_blocks: Option<u64> = None;
        let mut compact_db = false;
        let mut coin_maturity: Option<u64> = None;
        let mut restore = false;
        let mut debug_no_proofs = false;
        let mut args = args.into_iter().map(Into::into);

        while let Some(arg) = args.next() {
            let Some(s) = arg.to_str() else {
                return Err(ArgsError::Unknown(arg));
            };
            let (name, inline) = match s.split_once('=') {
                Some((name, value)) if name.starts_with("--") => (name, Some(value)),
                _ => (s, None),
            };
            match name {
                "-h" | "--help" => return flag("--help", inline, Command::Help),
                "-V" | "--version" => return flag("--version", inline, Command::Version),
                "--data-dir" => {
                    if data_dir.is_some() {
                        return Err(ArgsError::Duplicate("--data-dir"));
                    }
                    let value = match inline {
                        Some(v) => OsString::from(v),
                        None => args.next().ok_or(ArgsError::MissingValue("--data-dir"))?,
                    };
                    if value.is_empty() {
                        return Err(ArgsError::MissingValue("--data-dir"));
                    }
                    data_dir = Some(value.into());
                }
                "--no-mine"
                | "--no-listen"
                | "--status"
                | "--restore"
                | "--compact-db"
                | "--debug-no-proof-validation" => {
                    let (opt, set) = match name {
                        "--no-mine" => ("--no-mine", &mut no_mine),
                        "--no-listen" => ("--no-listen", &mut no_listen),
                        "--status" => ("--status", &mut status),
                        "--restore" => ("--restore", &mut restore),
                        "--compact-db" => ("--compact-db", &mut compact_db),
                        _ => ("--debug-no-proof-validation", &mut debug_no_proofs),
                    };
                    if inline.is_some() {
                        return Err(ArgsError::UnexpectedValue(opt));
                    }
                    if *set {
                        return Err(ArgsError::Duplicate(opt));
                    }
                    *set = true;
                }
                "--sync-depth" | "--snap-sync-limit" | "--keep-blocks" | "--coin-maturity" => {
                    let (opt, slot) = match name {
                        "--sync-depth" => ("--sync-depth", &mut sync_depth),
                        "--snap-sync-limit" => ("--snap-sync-limit", &mut snap_sync_limit),
                        "--keep-blocks" => ("--keep-blocks", &mut keep_blocks),
                        _ => ("--coin-maturity", &mut coin_maturity),
                    };
                    if slot.is_some() {
                        return Err(ArgsError::Duplicate(opt));
                    }
                    let value = match inline {
                        Some(v) => OsString::from(v),
                        None => args.next().ok_or(ArgsError::MissingValue(opt))?,
                    };
                    let blocks = value
                        .to_str()
                        .and_then(|v| v.parse().ok())
                        .ok_or(ArgsError::InvalidValue(opt))?;
                    *slot = Some(blocks);
                }
                "--max-peers" => {
                    if max_peers.is_some() {
                        return Err(ArgsError::Duplicate("--max-peers"));
                    }
                    let value = match inline {
                        Some(v) => OsString::from(v),
                        None => args.next().ok_or(ArgsError::MissingValue("--max-peers"))?,
                    };
                    let max = value
                        .to_str()
                        .and_then(|v| v.parse().ok())
                        .filter(|&n: &usize| n > 0)
                        .ok_or(ArgsError::InvalidValue("--max-peers"))?;
                    max_peers = Some(max);
                }
                "--listen" | "--peer" => {
                    let opt = if name == "--listen" {
                        "--listen"
                    } else {
                        "--peer"
                    };
                    if opt == "--listen" && listen.is_some() {
                        return Err(ArgsError::Duplicate("--listen"));
                    }
                    let value = match inline {
                        Some(v) => OsString::from(v),
                        None => args.next().ok_or(ArgsError::MissingValue(opt))?,
                    };
                    let addr = value
                        .to_str()
                        .and_then(|v| std::net::ToSocketAddrs::to_socket_addrs(v).ok()?.next())
                        .ok_or(ArgsError::InvalidValue(opt))?;
                    if opt == "--listen" {
                        listen = Some(addr);
                    } else {
                        peers.push(addr);
                    }
                }
                _ => return Err(ArgsError::Unknown(arg)),
            }
        }

        if no_listen && listen.is_some() {
            return Err(ArgsError::Conflict("--listen", "--no-listen"));
        }
        Ok(Command::Run(Args {
            data_dir: data_dir.unwrap_or_else(default_data_dir),
            no_mine,
            status,
            listen: (!no_listen).then(|| listen.unwrap_or(DEFAULT_LISTEN)),
            peers,
            sync_depth: sync_depth.unwrap_or(DEFAULT_SYNC_DEPTH),
            max_peers: max_peers.unwrap_or(DEFAULT_MAX_PEERS),
            snap_sync_limit: snap_sync_limit.unwrap_or(DEFAULT_SNAP_SYNC_LIMIT),
            keep_blocks: keep_blocks.unwrap_or(DEFAULT_KEEP_BLOCKS),
            compact_db,
            coin_maturity: coin_maturity.unwrap_or(DEFAULT_COIN_MATURITY),
            restore,
            debug_no_proofs,
        }))
    }
}

/// Result of a flag that takes no value and ends parsing.
fn flag(name: &'static str, inline: Option<&str>, cmd: Command) -> Result<Command, ArgsError> {
    match inline {
        Some(_) => Err(ArgsError::UnexpectedValue(name)),
        None => Ok(cmd),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Command, ArgsError> {
        Args::parse(args.iter().copied())
    }

    fn run(dir: impl Into<PathBuf>) -> Result<Command, ArgsError> {
        Ok(Command::Run(Args {
            data_dir: dir.into(),
            no_mine: false,
            status: false,
            listen: Some(DEFAULT_LISTEN),
            peers: vec![],
            sync_depth: DEFAULT_SYNC_DEPTH,
            max_peers: DEFAULT_MAX_PEERS,
            snap_sync_limit: DEFAULT_SNAP_SYNC_LIMIT,
            keep_blocks: DEFAULT_KEEP_BLOCKS,
            compact_db: false,
            coin_maturity: DEFAULT_COIN_MATURITY,
            restore: false,
            debug_no_proofs: false,
        }))
    }

    #[test]
    fn defaults() {
        assert_eq!(parse(&[]), run(default_data_dir()));
    }

    #[test]
    fn data_dir_forms() {
        assert_eq!(parse(&["--data-dir", "/tmp/x"]), run("/tmp/x"));
        assert_eq!(parse(&["--data-dir=/tmp/x"]), run("/tmp/x"));
        // Only the first '=' separates the value.
        assert_eq!(parse(&["--data-dir=a=b"]), run("a=b"));
        // A value that looks like an option is still taken as the value.
        assert_eq!(parse(&["--data-dir", "--help"]), run("--help"));
    }

    #[test]
    fn mode_flags() {
        let want = |no_mine, status| {
            Ok(Command::Run(Args {
                data_dir: default_data_dir(),
                no_mine,
                status,
                listen: Some(DEFAULT_LISTEN),
                peers: vec![],
                sync_depth: DEFAULT_SYNC_DEPTH,
                max_peers: DEFAULT_MAX_PEERS,
                snap_sync_limit: DEFAULT_SNAP_SYNC_LIMIT,
                keep_blocks: DEFAULT_KEEP_BLOCKS,
                compact_db: false,
                coin_maturity: DEFAULT_COIN_MATURITY,
                restore: false,
                debug_no_proofs: false,
            }))
        };
        assert_eq!(parse(&["--no-mine"]), want(true, false));
        assert_eq!(parse(&["--status"]), want(false, true));
        assert_eq!(parse(&["--status", "--no-mine"]), want(true, true));
    }

    #[test]
    fn sync_depth() {
        let depth = |args: &[&str]| match parse(args) {
            Ok(Command::Run(a)) => Ok(a.sync_depth),
            Ok(c) => panic!("{c:?}"),
            Err(e) => Err(e),
        };
        assert_eq!(depth(&[]), Ok(100));
        assert_eq!(depth(&["--sync-depth", "10"]), Ok(10));
        assert_eq!(depth(&["--sync-depth=0"]), Ok(0));
        use ArgsError::*;
        assert_eq!(
            depth(&["--sync-depth", "x"]),
            Err(InvalidValue("--sync-depth"))
        );
        assert_eq!(
            depth(&["--sync-depth", "-1"]),
            Err(InvalidValue("--sync-depth"))
        );
        assert_eq!(
            depth(&["--sync-depth", "1", "--sync-depth", "2"]),
            Err(Duplicate("--sync-depth"))
        );
    }

    #[test]
    fn keep_blocks_and_compaction() {
        let got = |args: &[&str]| match parse(args) {
            Ok(Command::Run(a)) => Ok((a.keep_blocks, a.compact_db)),
            Ok(c) => panic!("{c:?}"),
            Err(e) => Err(e),
        };
        assert_eq!(got(&[]), Ok((1000, false)));
        assert_eq!(got(&["--keep-blocks", "0", "--compact-db"]), Ok((0, true)));
        assert_eq!(got(&["--keep-blocks=5000"]), Ok((5000, false)));
        use ArgsError::*;
        assert_eq!(
            got(&["--keep-blocks", "x"]),
            Err(InvalidValue("--keep-blocks"))
        );
        assert_eq!(
            got(&["--compact-db", "--compact-db"]),
            Err(Duplicate("--compact-db"))
        );
    }

    #[test]
    fn snap_sync_limit() {
        let limit = |args: &[&str]| match parse(args) {
            Ok(Command::Run(a)) => Ok(a.snap_sync_limit),
            Ok(c) => panic!("{c:?}"),
            Err(e) => Err(e),
        };
        assert_eq!(limit(&[]), Ok(200));
        assert_eq!(limit(&["--snap-sync-limit", "50"]), Ok(50));
        assert_eq!(limit(&["--snap-sync-limit=0"]), Ok(0));
        use ArgsError::*;
        assert_eq!(
            limit(&["--snap-sync-limit", "-1"]),
            Err(InvalidValue("--snap-sync-limit"))
        );
        assert_eq!(
            limit(&["--snap-sync-limit"]),
            Err(MissingValue("--snap-sync-limit"))
        );
        assert_eq!(
            limit(&["--snap-sync-limit", "1", "--snap-sync-limit", "2"]),
            Err(Duplicate("--snap-sync-limit"))
        );
    }

    #[test]
    fn wallet_options() {
        let get = |args: &[&str]| match parse(args) {
            Ok(Command::Run(a)) => Ok((a.coin_maturity, a.restore)),
            Ok(c) => panic!("{c:?}"),
            Err(e) => Err(e),
        };
        assert_eq!(get(&[]), Ok((6, false)));
        assert_eq!(get(&["--coin-maturity", "10", "--restore"]), Ok((10, true)));
        assert_eq!(get(&["--coin-maturity=0"]), Ok((0, false)));
        use ArgsError::*;
        assert_eq!(
            get(&["--coin-maturity", "x"]),
            Err(InvalidValue("--coin-maturity"))
        );
        assert_eq!(
            get(&["--restore", "--restore"]),
            Err(Duplicate("--restore"))
        );
        assert_eq!(get(&["--restore=1"]), Err(UnexpectedValue("--restore")));
        let debug = |args: &[&str]| match parse(args) {
            Ok(Command::Run(a)) => a.debug_no_proofs,
            c => panic!("{c:?}"),
        };
        assert!(!debug(&[]) && debug(&["--debug-no-proof-validation"]));
    }

    #[test]
    fn max_peers() {
        let max = |args: &[&str]| match parse(args) {
            Ok(Command::Run(a)) => Ok(a.max_peers),
            Ok(c) => panic!("{c:?}"),
            Err(e) => Err(e),
        };
        assert_eq!(max(&[]), Ok(64));
        assert_eq!(max(&["--max-peers", "8"]), Ok(8));
        assert_eq!(max(&["--max-peers=200"]), Ok(200));
        use ArgsError::*;
        assert_eq!(max(&["--max-peers", "0"]), Err(InvalidValue("--max-peers")));
        assert_eq!(max(&["--max-peers", "x"]), Err(InvalidValue("--max-peers")));
        assert_eq!(max(&["--max-peers"]), Err(MissingValue("--max-peers")));
        assert_eq!(
            max(&["--max-peers", "1", "--max-peers", "2"]),
            Err(Duplicate("--max-peers"))
        );
    }

    #[test]
    fn network_options() {
        let a: SocketAddr = "127.0.0.1:7333".parse().unwrap();
        let b: SocketAddr = "10.0.0.2:9".parse().unwrap();
        let want = Ok(Command::Run(Args {
            data_dir: default_data_dir(),
            no_mine: false,
            status: false,
            listen: Some(a),
            peers: vec![b, a],
            sync_depth: DEFAULT_SYNC_DEPTH,
            max_peers: DEFAULT_MAX_PEERS,
            snap_sync_limit: DEFAULT_SNAP_SYNC_LIMIT,
            keep_blocks: DEFAULT_KEEP_BLOCKS,
            compact_db: false,
            coin_maturity: DEFAULT_COIN_MATURITY,
            restore: false,
            debug_no_proofs: false,
        }));
        assert_eq!(
            parse(&[
                "--listen",
                "127.0.0.1:7333",
                "--peer=10.0.0.2:9",
                "--peer",
                "127.0.0.1:7333"
            ]),
            want
        );
        use ArgsError::*;
        assert_eq!(
            parse(&["--listen", "nowhere"]),
            Err(InvalidValue("--listen"))
        );
        assert_eq!(parse(&["--peer"]), Err(MissingValue("--peer")));
        assert_eq!(
            parse(&["--listen", "127.0.0.1:1", "--listen", "127.0.0.1:2"]),
            Err(Duplicate("--listen"))
        );
        // Listening is the default; --no-listen turns it off.
        let listen = |args: &[&str]| match parse(args) {
            Ok(Command::Run(a)) => Ok(a.listen),
            Ok(c) => panic!("{c:?}"),
            Err(e) => Err(e),
        };
        assert_eq!(listen(&[]), Ok(Some("0.0.0.0:3737".parse().unwrap())));
        assert_eq!(listen(&["--no-listen"]), Ok(None));
        assert_eq!(
            listen(&["--no-listen", "--listen", "127.0.0.1:1"]),
            Err(Conflict("--listen", "--no-listen"))
        );
    }

    #[test]
    fn help_and_version() {
        for a in ["-h", "--help"] {
            assert_eq!(parse(&[a]), Ok(Command::Help));
        }
        for a in ["-V", "--version"] {
            assert_eq!(parse(&[a]), Ok(Command::Version));
        }
        assert_eq!(parse(&["--data-dir", "x", "--help"]), Ok(Command::Help));
        assert_eq!(parse(&["--help", "--bogus"]), Ok(Command::Help));
    }

    #[test]
    fn errors() {
        use ArgsError::*;
        assert_eq!(parse(&["--data-dir"]), Err(MissingValue("--data-dir")));
        assert_eq!(parse(&["--data-dir="]), Err(MissingValue("--data-dir")));
        assert_eq!(
            parse(&["--data-dir", "a", "--data-dir", "b"]),
            Err(Duplicate("--data-dir"))
        );
        assert_eq!(parse(&["--help=1"]), Err(UnexpectedValue("--help")));
        assert_eq!(parse(&["--no-mine=1"]), Err(UnexpectedValue("--no-mine")));
        assert_eq!(parse(&["--status", "--status"]), Err(Duplicate("--status")));
        assert_eq!(
            parse(&["--dev-chain", "5"]),
            Err(Unknown("--dev-chain".into()))
        );
        assert_eq!(parse(&["--bogus"]), Err(Unknown("--bogus".into())));
        assert_eq!(parse(&["-x=1"]), Err(Unknown("-x=1".into())));
        assert_eq!(parse(&["stray"]), Err(Unknown("stray".into())));
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_data_dir() {
        use std::os::unix::ffi::OsStringExt;
        let raw = OsString::from_vec(vec![b'd', 0xff]);
        let parsed = Args::parse([OsString::from("--data-dir"), raw.clone()]);
        assert_eq!(
            parsed,
            Ok(Command::Run(Args {
                data_dir: raw.into(),
                no_mine: false,
                status: false,
                listen: Some(DEFAULT_LISTEN),
                peers: vec![],
                sync_depth: DEFAULT_SYNC_DEPTH,
                max_peers: DEFAULT_MAX_PEERS,
                snap_sync_limit: DEFAULT_SNAP_SYNC_LIMIT,
                keep_blocks: DEFAULT_KEEP_BLOCKS,
                compact_db: false,
                coin_maturity: DEFAULT_COIN_MATURITY,
                restore: false,
                debug_no_proofs: false,
            }))
        );
    }
}
