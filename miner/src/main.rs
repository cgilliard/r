use std::process::ExitCode;

use r::args::{Args, Command, USAGE};

fn main() -> ExitCode {
    let args = match Args::from_env() {
        Ok(Command::Run(args)) => args,
        Ok(Command::Help) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Ok(Command::Version) => {
            println!("r {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let _ = args;
    ExitCode::SUCCESS
}
