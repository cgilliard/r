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
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.status {
        return r::daemon::status(&args.data_dir.join("chain"));
    }
    let net = r::daemon::Net {
        listen: args.listen,
        peers: args.peers.clone(),
        sync_depth: args.sync_depth,
        max_peers: args.max_peers,
        snap_sync_limit: args.snap_sync_limit,
    };
    let opts = r::daemon::Options {
        mine: !args.no_mine,
        coin_maturity: args.coin_maturity,
        restore: args.restore,
    };
    r::daemon::run(&args.data_dir, &net, &opts)
}
