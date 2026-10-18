use std::process::ExitCode;

use r::args::{Args, Command, USAGE};
use r::{db, state::StateTree};

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
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let env = db::open_env(&args.data_dir)?;
    let tree = StateTree::open(&env)?;
    let state = tree.state(&env.read_txn()?)?;
    println!("state: {} outputs", state.size - 1);
    Ok(())
}
