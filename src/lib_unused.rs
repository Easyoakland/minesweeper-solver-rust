/* use scrap::{Capturer, Display};
use std::io::ErrorKind::WouldBlock;
use std::time::Duration;
use std::thread; */

/*
/// Returns a capturer for the specified display. Used for scrap.
 pub fn setup_capturer(id: usize) -> Capturer {
    // Get capturer for capturing frames from screen.
    let mut displays = Display::all().expect("Couldn't find primary display.");
    let display = displays.swap_remove(id);
    let capturer = Capturer::new(display).expect("Couldn't begin capture.");
    return capturer;
} */

/*
/// Returns a vector of concatenated RGBA values corresponding to a captured frame. Uses scrap.
 pub fn capture_frame_scrap<'a>(
    capturer: &mut Capturer,
    one_frame: Duration,
    width: usize,
    height: usize,
) -> Vec<u8> {
    loop {
        // Wait until there's a frame.
        let buffer = match capturer.frame() {
            Ok(buffer) => buffer,
            Err(error) => {
                if error.kind() == WouldBlock {
                    // Keep spinning.
                    // println!("Waiting...");
                    thread::sleep(one_frame);
                    continue;
                } else {
                    panic!("Error: {}", error);
                }
            }
        };

        // Make sure the image is not a failed black screen.
        if !buffer.to_vec().iter().any(|&x| x != 0) {
            thread::sleep(Duration::new(0, 1)); // sleep 1ms
            continue;
        }

        // Flip the BGRA image into a RGBA image
        let mut bitflipped = Vec::with_capacity(width * height * 4);
        let stride = buffer.len() / height;

        for y in 0..height {
            for x in 0..width {
                let i = stride * y + 4 * x;
                bitflipped.extend_from_slice(&[buffer[i + 2], buffer[i + 1], buffer[i], 255]);
            }
        }

        return bitflipped;
    }
}
 */
