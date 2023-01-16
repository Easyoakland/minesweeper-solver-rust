use minesweeper_solver_in_rust::{capture_frame, save_rgb_frame, setup_capturer};

fn main() {
    let mut capturer = setup_capturer(0);

    let (width, height) = capturer.geometry();


}
