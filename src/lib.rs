use captrs::{Bgr8, Capturer};
use image::{Rgb, RgbImage, Rgba, RgbaImage};

/// Returns a capturer instance. Selects monitor based upon passed id zero indexed.
pub fn setup_capturer(id: usize) -> Capturer {
    return Capturer::new(id).unwrap();
}

/// Returns a vector of concatenated RGB values corresponding to a captured frame.
pub fn capture_frame(capturer: &mut Capturer) -> Vec<u8> {
    loop {
        let temp = capturer.capture_frame();
        match temp {
            Ok(ps) => {
                let mut rgb_vec = Vec::new();
                for Bgr8 {
                    r,
                    g,
                    b, /* a */
                    ..
                } in ps.into_iter()
                {
                    rgb_vec.push(r);
                    rgb_vec.push(g);
                    rgb_vec.push(b);
                    /* rgb_vec.push(a); */
                }

                // Make sure the image is not a failed black screen.
                if !rgb_vec.iter().any(|&x| x != 0) {
                    // thread::sleep(Duration::new(0, 1)); // sleep 1ms
                    // println!("All black");
                    continue;
                };
                return rgb_vec;
            }
            Err(_) => continue,
        }
    }
}

/// Expects buffer to be a RGBA image.
pub fn save_rgba_frame(path: &str, buffer: Vec<u8>, width: usize, height: usize) {
    let stride = buffer.len() / height;

    let mut img = RgbaImage::new(width as u32, height as u32);
    for x in 0..width {
        for y in 0..height {
            let i = stride * y + 4 * x;
            img.put_pixel(
                x as u32,
                y as u32,
                Rgba([buffer[i], buffer[i + 1], buffer[i + 2], buffer[i + 3]]),
            );
        }
    }
    img.save(path)
        .expect("Should have been able to save image to path.");
}

/// Expects buffer to be a RGB image.
pub fn save_rgb_frame(path: &str, buffer: Vec<u8>, width: usize, height: usize) {
    let stride = buffer.len() / height;

    let mut img = RgbImage::new(width as u32, height as u32);
    for x in 0..width {
        for y in 0..height {
            let i = stride * y + 3 * x;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// This test requires manual confirmation.
    #[test]
    fn record_screen_to_file() {
        let mut capturer = setup_capturer(0);
        let (width, height) = capturer.geometry();
        for i in 0..=5 {
            let rgb_vec = capture_frame(&mut capturer);
            let path = format!("test/IMG{i}.png");
            save_rgb_frame(&path, rgb_vec, width as usize, height as usize);
            // assert_eq!(std::path::Path::new(&path).exists(), true); // Check file now exists.
        }
    }
}
