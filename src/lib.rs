use scrap::{Capturer, Display};
use std::io::ErrorKind::WouldBlock;
use std::time::Duration;
use std::thread;
use image::{Rgb, RgbImage};


/// Returns a vector of concatenated RGBA values.
#[inline]
pub fn capture_frame<'a>(
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

/// Expects buffer to be a RGBA image.
pub fn save_frame(path: &str, buffer: Vec<u8>, width: usize, height: usize) {
    let stride = buffer.len() / height;

    let mut img = RgbImage::new(width as u32, height as u32);
    for x in 0..width {
        for y in 0..height {
            let i = stride * y + 4 * x;
            img.put_pixel(
                x as u32,
                y as u32,
                Rgb([buffer[i], buffer[i + 1], buffer[i + 2]]),
            );
        }
    }
    img.save(path)
        .expect("Should have been able to save image to path.");
}

/// Returns a capturer for the specified display.
pub fn setup_capturer(id: usize) -> Capturer{
    // Get capturer for capturing frames from screen.
    let mut displays = Display::all().expect("Couldn't find primary display.");
    let display = displays.swap_remove(id);
    let capturer = Capturer::new(display).expect("Couldn't begin capture.");
    return capturer;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}