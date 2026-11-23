//! `frenz`: scan for, record from, summarize and export recordings of FRENZ brainbands.

fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(frenz_cli::run(std::env::args_os()))
}
