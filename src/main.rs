// #![windows_subsystem = "windows"]

use minesweeper_solver_in_rust::*;
// use scrap::{Capturer, Display};
use std::time::Duration;

fn main() {
    let one_second = Duration::new(1, 0);
    let one_frame = one_second / 60;

    let mut capturer = setup_capturer(0);
    let (width, height) = (capturer.width(), capturer.height());

    // let mut frame = capture_frame(&mut capturer, one_frame, width, height);

    for i in 0..20 {
        let frame = capture_frame(&mut capturer, one_frame, width, height);
        let path = format!("IMG{i}.png");
        save_frame(&path, frame, width, height);
    }
    // Save the image.
}
