use captrs::{Bgr8, Capturer};
use image::{imageops, io, DynamicImage, GenericImageView, /* ImageBuffer, */ Rgb, RgbImage};
/* use imageproc::rgb_image; */
use enigo::{Enigo, MouseControllable};
use std::io::prelude::*;
use std::{
    fs::OpenOptions,
    ops::{Add, Sub},
};

const TIMEOUTS_ATTEMPTS_NUM: u8 = 10;
const MAX_COMBINATIONS: u32 = 2000000;

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

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct CellCord(pub usize, pub usize);

impl Add for CellCord {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1)
    }
}

impl Sub for CellCord {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0, self.1 - other.1)
    }
}

/// Returns a capturer instance. Selects monitor based upon passed id. Zero indexed.
pub fn setup_capturer(id: usize) -> Capturer {
    return Capturer::new(id).unwrap();
}

/// Returns a vector of concatenated RGB values corresponding to a captured frame.
fn capture_rgb_frame(capturer: &mut Capturer) -> Vec<u8> {
    loop {
        let temp = capturer.capture_frame();
        match temp {
            Ok(frame) => {
                let mut rgb_vec = Vec::new();
                for Bgr8 {
                    r,
                    g,
                    b, /* a */
                    ..
                } in frame.into_iter()
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

/// Saves an RGB image vector to a the given path.
pub fn save_rgb_vector(path: &str, buffer: Vec<u8>, width: u32, height: u32) {
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

/// Returns a vector containing the top left cordinate of each instance of the subimage found in the super image. Matches with rgb8.
/// Should always find locations of the sub image in the order from left to right then top to bottom in the super image.
pub fn locate_all(sup_image: &DynamicImage, sub_image: &DynamicImage) -> Vec<Point> {
    let sup_image = sup_image.clone().into_rgb8(); // `.clone().into_rgb8()` is 13% faster on benchmark then `.to_rgb8()`.
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
            let sub_sup_image = imageops::crop_imm(&sup_image, x, y, sub_width, sub_height);

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
            for (sup_data, sub_data) in sub_sup_image
                .pixels()
                .map(|(_x, _y, pixel)| pixel)
                .zip(sub_image.pixels())
            {
                if sub_data != &sup_data {
                    continue 'x_loop;
                }
            }

            output.push(Point(x as i32, y as i32));
        }
    }
    return output;
}

const CELL_VARIANT_COUNT: usize = 11;
#[derive(Debug, Copy, Clone, PartialEq)]
enum CellKind {
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Flag,
    Unexplored,
    Explored,
}

/// Holds the information related to the current state of the game.
pub struct Game {
    cell_images: Vec<RgbImage>,
    state: Vec<CellKind>,
    cell_positions: Vec<Point>,
    px_width: u32,
    px_height: u32,
    pub cell_width: u32,
    pub cell_height: u32,
    top_left: Point,
    bottom_right: Point,
    individual_cell_width: u32,
    individual_cell_height: u32,
    board_screenshot: RgbImage,
    capturer: Capturer,
}

impl Game {
    /// Creates a new game struct. Requires the input of the different cell type image file locations in order of CELL enum variants.
    /// # Panics
    /// - If the given files are not readable and decodable it panics.
    /// - If a board is unable to be found it panics.
    pub fn new(cell_files: [&str; CELL_VARIANT_COUNT], id: usize) -> Game {
        // Reads in each cell image.
        let mut cell_images: Vec<RgbImage> = Vec::with_capacity(CELL_VARIANT_COUNT);
        for file in cell_files.iter() {
            cell_images.push(
                io::Reader::open(file)
                    .expect(
                        format!("Cell variant image \"{file}\" was unable to be read.").as_str(),
                    )
                    .decode()
                    .expect(format!("Unsupported Image type for \"{file}\".").as_str())
                    .to_rgb8(),
            );
        }

        let mut capturer = setup_capturer(id);

        // Finds the initial positions of the cells in the game grid.
        let screenshot = capture_image_frame(&mut capturer);
        /* save_rgb_vector("screenshot.png", screenshot.to_vec(), screenshot.width(), screenshot.height());
        save_rgb_vector("cell_at_0.png", cell_images[9].to_vec(), cell_images[9].width(), cell_images[9].height()); */
        let cell_positions = locate_all(
            &image::DynamicImage::from(screenshot.clone()),
            &image::DynamicImage::from(cell_images[9].clone()),
        );

        // If no grid was found then it is not possible to continue.
        if cell_positions.len() == 0 {
            panic!("Unable to find board. Unable to continue.");
        }

        // Initializes state as all unexplored.
        let state = vec![CellKind::Unexplored; cell_positions.len()];

        let top_left = cell_positions[0];

        let individual_cell_width = cell_images[9].width();
        let individual_cell_height = cell_images[9].height();

        // Setting the bottom right requires finding hte top left of the last cell and shifting by the cell's width and height to get to the bottom right.
        let bottom_right = cell_positions
            .last()
            .expect("Will only fail if there is no game grid with cells");
        let offset_from_top_left_corner_to_bottom_right_corner_of_cell =
            Point(individual_cell_width as i32, individual_cell_height as i32);
        // Update bottom_right by offset to go from top left to bottom right of last cell.
        let bottom_right =
            *bottom_right + offset_from_top_left_corner_to_bottom_right_corner_of_cell;

        let mut biggest: i32 = 0; // Temp variable for holding the largest position in the following iterations.
        let mut cell_width = 0;
        // Set the width by counting how many cells into cell_positions the highest x value is.
        // This only works if since the grid is square because this counts the width of the first row.
        for (i, cell_point) in cell_positions.iter().enumerate() {
            if cell_point.0 > biggest {
                biggest = cell_point.0;
                cell_width = i + 1;
            }
        }
        let mut cell_height = 0;
        // Set the height by counting how many widths into cell_positions the highest y value is.
        biggest = 0;
        for (i, cell_point) in cell_positions.iter().step_by(cell_width).enumerate() {
            if cell_point.1 > biggest {
                biggest = cell_point.1;
                cell_height = i + 1;
            }
        }

        // The width in pixels is the number of cells times the width of each cell.
        let px_width = (cell_width as u32) * cell_images[9].width();
        // Same for height.
        let px_height = (cell_height as u32) * cell_images[9].height();

        return Game {
            cell_images,
            state,
            cell_positions,
            px_width: px_width as u32,
            px_height: px_height as u32,
            cell_width: cell_width as u32,
            cell_height: cell_height as u32,
            top_left,
            bottom_right,
            individual_cell_width,
            individual_cell_height,
            capturer,
            board_screenshot: screenshot,
        };
    }

    fn click(&self, cord: CellCord) {
        let mut enigo = Enigo::new();
        let x = self.cell_positions[cord.0].0 + 1;
        let y = self.cell_positions[self.individual_cell_height as usize * cord.1].1 + 1;
        enigo.mouse_move_to(x, y);
        enigo.mouse_down(enigo::MouseButton::Left);
        enigo.mouse_up(enigo::MouseButton::Left);
    }

    /* pub fn reveal (&mut self, cord: CellCord) {
        self.click(cord);
        let a = CellKind::Unexplored;
        let i = 0;
        while a == CellKind::Unexplored  // This will keep looking for a change until one occurs.
            {if i >= TIMEOUTS_ATTEMPTS_NUM  // If it must wait this many times give up because something is wrong.
                {
                panic!("Board won't update. Game Loss?");}
            self.set_board_screenshot();
            a = self.identify_cell(CellCord);
            // If the clicked cell looks like it is an unnumbered and uncovered cell then check that it's neighbors aren't uncovered.
            // If they are all uncovered then the clicked cell is not finished loading and it only looks this way because that's how the program is displaying it temporarily while it loads.
            if a == CellKind::Explored{
                // make a Cell Object for the clicked cell;
                let clicked_cell = Cell(CellCord, "complete.png");
                // iterate through the neighbors of the clicked_cell;
                    'neighbor_loop: for neighbor in clicked_cell.neighbors(1, self.cell_width, self.cell_height){
                        if self.identify_cell(neighbor) == CellKind::Unexplored{
                            // If any neighbors are unexplored then this cell can't be a blank unexplored
                            a = CellKind::Unexplored;  // set a so while loop will take a new screenshot;
                            break 'neighbor_loop;}}
            i += 1;}
        self.update_state(cord);
    } */

    /*         fn identify_cell(&self, cord: CellCord) {
        // Only unchecked cells can update so don't bother checking if it wasn't an unchecked cell last time it was updated.
        let temp = self.recall_CellKind(cord);
        if temp != CellKind::Unexplored{
            return temp;}
        let pos = self.convert_CellCord_to_pos(cord);
        // return the first tiletype that matches the tile at the given cordinate
        for (i, cellTypeIm) in self.cellTypeIms.iters().enumerate() {
            // this works by computing the pixel by pixel difference between the image on the board and a reference image.
            dif = ImageChops.difference(cellTypeIm, self.boardIm.crop(
                (pos[0]-self._origin[0], pos[1]-self._origin[1], pos[0]-self._origin[0]+self._cellwidth, pos[1]-self._origin[1]+self._cellheight)))
            // if the sum difference of every pixel is zero then they are the same image
            if sum(sum(sum(np.asarray(dif)))) == 0{
                temp = self.cellTypes[i];
                return temp;}}
        println!("UNIDENTIFIED CELL at cord: " + str(cord));

        // If the program can't identify the cell then it shouldn't keep trying to play the game.
        exiting(self)
    } */

    fn set_board_screenshot(&mut self) {
        let screenshot = capture_image_frame(&mut self.capturer);
        self.board_screenshot = image::imageops::crop_imm(
            &screenshot,
            self.top_left.0 as u32,
            self.top_left.1 as u32,
            self.px_width,
            self.px_height,
        )
        .to_image();
    }

    /// Saves information about this Game to file for potential debugging purposes. If panic = true then it panics after doing so.
    pub fn exit(&self, panic: bool) {
        // save saved game state as picture
        // TODO reimplement in rust // game.showGameSavedState().save("FinalGameState.png");

        // save game state as csv
        // clear file contents
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open("test/FinalGameState.csv")
            .unwrap();
        if let Err(e) = write!(file, "") {
            eprintln!("Couldn't write to file: {}", e);
        }
        // Write state to file after formatting nicely.
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open("test/FinalGameState.csv")
            .unwrap();

        for (i, cell) in self.state.iter().enumerate() {
            if let Err(e) = write!(file, "{cell:?}, ") {
                eprintln!("Couldn't write to file: {}", e);
            }
            if (i + 1) % self.cell_width as usize == 0 {
                if let Err(e) = write!(file, "\n") {
                    eprintln!("Couldn't write to file: {}", e);
                }
            }
        }
        if panic == true {
            panic!("Exit functions was told to panic.");
        }
    }
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
            save_rgb_vector(&path, rgb_vec, width, height);
            // assert_eq!(std::path::Path::new(&path).exists(), true); // Check file now exists.
        }
    }

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

    /// This test requires manual confirmation.
    #[ignore]
    #[test]
    fn click_1_1() {
        let game = Game::new(
            [
                "cell_images/1.png",
                "cell_images/2.png",
                "cell_images/3.png",
                "cell_images/4.png",
                "cell_images/5.png",
                "cell_images/6.png",
                "cell_images/7.png",
                "cell_images/8.png",
                "cell_images/flag.png",
                "cell_images/cell.png",
                "cell_images/complete.png",
            ],
            0,
        );

        game.click(CellCord(1, 1));
    }

    /// This test requires manual confirmation.
    #[ignore]
    #[test]
    fn set_board_screenshot_test() {
        let mut game = Game::new(
            [
                "cell_images/1.png",
                "cell_images/2.png",
                "cell_images/3.png",
                "cell_images/4.png",
                "cell_images/5.png",
                "cell_images/6.png",
                "cell_images/7.png",
                "cell_images/8.png",
                "cell_images/flag.png",
                "cell_images/cell.png",
                "cell_images/complete.png",
            ],
            0,
        );
        game.set_board_screenshot();
        save_rgb_vector(
            "board_screenshot_test.png",
            game.board_screenshot.to_vec(),
            game.board_screenshot.width(),
            game.board_screenshot.height(),
        );
    }
}
