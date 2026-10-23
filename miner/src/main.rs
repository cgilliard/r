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
    let chain = args.data_dir.join("chain");
    if args.compact_db && chain.join("data.mdb").exists() {
        println!("compacting the database in {}...", chain.display());
        let (before, after) = r::db::compact(&chain)?;
        println!(
            "compacted: {:.1} MB -> {:.1} MB",
            before as f64 / 1e6,
            after as f64 / 1e6
        );
    }
    let net = r::daemon::Net {
        listen: args.listen,
        peers: args.peers.clone(),
        sync_depth: args.sync_depth,
        max_peers: args.max_peers,
        snap_sync_limit: args.snap_sync_limit,
        keep_blocks: args.keep_blocks,
        slate_hosts: args.slate_hosts.clone(),
    };
    let opts = r::daemon::Options {
        mine: !args.no_mine,
        coin_maturity: args.coin_maturity,
        restore: args.restore,
        debug_no_proofs: args.debug_no_proofs,
        auto_receive: !args.no_automatic_receive,
    };
    r::daemon::run(&args.data_dir, &net, &opts)
}
