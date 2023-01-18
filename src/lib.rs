use captrs::{Bgr8, Capturer};
use image::{imageops, DynamicImage, /* ImageBuffer, Rgb, */ RgbImage};
/* use imageproc::rgb_image; */
use std::ops::{Add, Sub};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Point(pub i32, pub i32);

impl Add for Point {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1)
    }
}

impl Sub for Point {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0, self.1 - other.1)
    }
}

/// Returns a capturer instance. Selects monitor based upon passed id zero indexed.
pub fn setup_capturer(id: usize) -> Capturer {
    return Capturer::new(id).unwrap();
}

/// Returns a vector of concatenated RGB values corresponding to a captured frame.
fn capture_rgb_frame(capturer: &mut Capturer) -> Vec<u8> {
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

/// Captures and returns a screenshot as RgbImage.
pub fn capture_image_frame(capturer: &mut Capturer) -> RgbImage {
    return RgbImage::from_raw(
        capturer.geometry().0,
        capturer.geometry().1,
        capture_rgb_frame(capturer),
    )
    .expect("Frame was unable to be captured and converted to RGB.");
}

/// Expects buffer to be a RGB image.
pub fn save_rgb_frame(path: &str, buffer: Vec<u8>, width: u32, height: u32) {
    let img =
        RgbImage::from_raw(width, height, buffer).expect("Couldn't convert buffer to rgb image.");
    image::save_buffer(
        path,
        &img,
        img.width(),
        img.height(),
        image::ColorType::Rgb8,
    )
    .expect("Couldn't save buffer.");
}

/// Returns a vector contains the top left cordinate of each instance of the subimage found in the superimage
pub fn locate_all(sup_image: &DynamicImage, sub_image: &DynamicImage) -> Vec<Point> {
    let sup_image = sup_image.clone().into_rgb8();
    let sub_image = sub_image.clone().into_rgb8();
    let mut output = Vec::new();

    // Get dimensions of both images.
    let (sup_width, sup_height) = (sup_image.width(), sup_image.height());
    let (sub_width, sub_height) = (sub_image.width(), sub_image.height());

    // Iterate through all positions of the sup_image checking if that region matches the sub_image when cropped to size.
    'y_loop: for y in 0..sup_height {
        'x_loop: for x in 0..sup_width {
            // Skip to next y line if no space left on this row for a sub image to fit.
            if x + sub_width > sup_width {
                continue 'y_loop;
            }
            // Stop searching if no room for any more images.
            if y + sub_height > sup_height {
                break 'y_loop;
            }
            // Generate the cropped image.
            let sub_sup_image =
                imageops::crop_imm(&sup_image, x, y, sub_width, sub_height).to_image();

            // Following code is used for debugging. It will print the sub image and section of super image being examined.
            /*image::save_buffer(
                "sub_sup_image.png",
                &sub_sup_image,
                sub_sup_image.width(),
                sub_sup_image.height(),
                image::ColorType::Rgb8,
            ).expect("Can't save sup_sup_image");
            image::save_buffer(
                "sub_image.png",
                &sub_image,
                sub_image.width(),
                sub_image.height(),
                image::ColorType::Rgb8,
            ).expect("Can't save sup_sup_image"); */

            // For each point if it doesn't match skip to next position.
            for (sup_data, sub_data) in sub_sup_image.iter().zip(sub_image.iter()) {
                if sub_data != sup_data {
                    continue 'x_loop;
                }
            }

            output.push(Point(x as i32, y as i32));
        }
    }
    return output;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This test requires manual confirmation.
    #[ignore]
    #[test]
    fn must_check_output_manually_record_screen_to_file() {
        let mut capturer = setup_capturer(0);
        let (width, height) = capturer.geometry();
        for i in 0..=5 {
            let rgb_vec = capture_rgb_frame(&mut capturer);
            let path = format!("test/IMG{i}.png");
            save_rgb_frame(&path, rgb_vec, width, height);
            // assert_eq!(std::path::Path::new(&path).exists(), true); // Check file now exists.
        }
    }

    use image::io;
    #[test]
    fn test_small_board_sub_image_search() {
        let super_image = io::Reader::open("test_in/subimage_search/sub_board.png")
            .expect("Couldn't read super image.")
            .decode()
            .expect("Unsupported Type");
        let sub_image = io::Reader::open("test_in/subimage_search/cell.png")
            .expect("Couldn't read sub image")
            .decode()
            .expect("Unsupported Type");
        let all_positions = locate_all(&super_image, &sub_image);
        assert!(all_positions.iter().any(|point| *point == Point(10, 48)));
        assert!(all_positions.iter().any(|point| *point == Point(26, 48)));
        assert_eq!(all_positions.len(), 2);
    }

    #[test]
    fn test_board_sub_image_search() {
        let super_image = io::Reader::open("test_in/subimage_search/board.png")
            .expect("Couldn't read super image.")
            .decode()
            .expect("Unsupported Type");
        let sub_image = io::Reader::open("test_in/subimage_search/cell.png")
            .expect("Couldn't read sub image")
            .decode()
            .expect("Unsupported Type");
        let all_positions = locate_all(&super_image, &sub_image);
        assert!(all_positions.iter().any(|point| *point == Point(10, 48)));
        assert!(all_positions.iter().any(|point| *point == Point(26, 48)));
        assert_eq!(all_positions.len(), 16 * 30);
    }
}
