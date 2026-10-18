//! Command-line argument parsing.

use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;

pub const USAGE: &str = "\
Usage: r [OPTIONS]

Options:
      --data-dir <PATH>  Directory for the database [default: data]
  -h, --help             Print help
  -V, --version          Print version";

pub const DEFAULT_DATA_DIR: &str = "data";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Args {
    pub data_dir: PathBuf,
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
    Unknown(OsString),
}

impl fmt::Display for ArgsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingValue(opt) => write!(f, "option '{opt}' requires a value"),
            Self::Duplicate(opt) => write!(f, "option '{opt}' was given more than once"),
            Self::UnexpectedValue(opt) => write!(f, "option '{opt}' does not take a value"),
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
                _ => return Err(ArgsError::Unknown(arg)),
            }
        }

        Ok(Command::Run(Args {
            data_dir: data_dir.unwrap_or_else(|| DEFAULT_DATA_DIR.into()),
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

    fn run(dir: &str) -> Result<Command, ArgsError> {
        Ok(Command::Run(Args {
            data_dir: dir.into(),
        }))
    }

    #[test]
    fn defaults() {
        assert_eq!(parse(&[]), run(DEFAULT_DATA_DIR));
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
                data_dir: raw.into()
            }))
        );
    }
}
