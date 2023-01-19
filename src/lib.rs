use captrs::{Bgr8, Capturer};
use image::{imageops, io, DynamicImage, GenericImageView, /* ImageBuffer,  Rgb,*/ RgbImage};
/* use imageproc::rgb_image; */
use enigo::{Enigo, MouseControllable};
use enum_iterator::{all, Sequence};
use std::io::prelude::*;
use std::{
    error::Error,
    fmt,
    fs::OpenOptions,
    ops::{Add, Sub},
};

// TODO replace "as usize" with .try_into().unwrap()
// TODO change usize operations with checked_sub or checked_add etc..

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

#[derive(Debug)]
struct GameError(&'static str);

/// Error that indicates something went wrong while trying to solve the game.
impl Error for GameError {}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/* impl fmt::Debug for GameError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{ file: {}, line: {} }}", file!(), line!())
    }
} */

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
    let sup_image = sup_image.clone().into_rgb8(); // `.clone().into_rgb8()` is 13% faster on benchmark then `.to_rgb8()`. As approx as fast.as_rgb.unwrap() (maybe 1% faster) but can use all of RgbImage methods this way.
    let sub_image = sub_image.clone().into_rgb8();
    let mut output = Vec::new();

    // Get dimensions of both images.
    let (sup_width, sup_height) = (sup_image.width(), sup_image.height());
    let (sub_width, sub_height) = (sub_image.width(), sub_image.height());

    // Iterate through all positions of the sup_image checking if that region matches the sub_image when cropped to size.
    let mut sub_sup_image = imageops::crop_imm(&sup_image, 0, 0, sub_width, sub_height);
    'y_loop: for y in 0..sup_height {
        'x_loop: for x in 0..sup_width {
            // Move sub image instead of creating a new one each iteration.
            // Approx 10% faster than using `let mut sub_sup_image = imageops::crop_imm(&sup_image, x, y, sub_width, sub_height);` each line
            sub_sup_image.change_bounds(x, y, sub_width, sub_height);

            // Skip to next y line if no space left on this row for a sub image to fit.
            if x + sub_width > sup_width {
                continue 'y_loop;
            }
            // Stop searching if no room for any more images.
            if y + sub_height > sup_height {
                break 'y_loop;
            }

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

fn exact_image_match(image1: &DynamicImage, image2: &DynamicImage) -> bool {
    // If the images don't have the same dimensions then they can't be an exact match.
    if image1.width() != image2.width() {
        return false;
    } else if image1.height() != image2.height() {
        return false;
    };

    // If they are the same dimension then locating all of either in the other will give one result if they are the same and 0 if different.
    let matches = locate_all(image1, image2);
    if matches.len() == 0 {
        return false;
    } else if matches.len() == 1 {
        return true;
    } else {
        panic!("Somehow one image exists multiple times in another image of the same size. Something is wrong.")
    }
}

const CELL_VARIANT_COUNT: usize = 11;
#[derive(Debug, Copy, Clone, PartialEq, Sequence)]
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
    board_px_width: u32,
    board_px_height: u32,
    board_cell_width: u32,
    board_cell_height: u32,
    top_left: Point,
    bottom_right: Point,
    individual_cell_width: u32,
    individual_cell_height: u32,
    board_screenshot: RgbImage,
    capturer: Capturer,
    frontier: Vec<Cell>,
}

impl Game {
    /// Creates a new game struct. Requires the input of the different cell type image file locations in order of CELL enum variants.
    /// # Panics
    /// - If the given files are not readable and decodable it panics.
    /// - If a board is unable to be found it panics.
    pub fn build(
        cell_files: [&str; CELL_VARIANT_COUNT],
        screenshot: RgbImage,
        capturer: Capturer,
    ) -> Game {
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
            board_px_width: px_width as u32,
            board_px_height: px_height as u32,
            board_cell_width: cell_width as u32,
            board_cell_height: cell_height as u32,
            top_left,
            bottom_right,
            individual_cell_width,
            individual_cell_height,
            capturer,
            board_screenshot: screenshot, // Initially not cropped to just board. Will be when it is set using the correct method.
            frontier: Vec::new(),
        };
    }

    pub fn new(cell_files: [&str; CELL_VARIANT_COUNT], id: usize) -> Game {
        let mut capturer = setup_capturer(id);

        // Finds the initial positions of the cells in the game grid.
        let screenshot = capture_image_frame(&mut capturer);
        return Game::build(cell_files, screenshot, capturer);
    }

    /// Sets the board screenshot of Game to just the board of tiles. Crops out extra stuff.
    fn get_board_screenshot_from_screen(&mut self) {
        let screenshot = capture_image_frame(&mut self.capturer);
        self.board_screenshot = image::imageops::crop_imm(
            &screenshot,
            self.top_left.0 as u32,
            self.top_left.1 as u32,
            self.board_px_width,
            self.board_px_height,
        )
        .to_image();
    }

    fn click(&self, cord: CellCord) {
        let mut enigo = Enigo::new();
        // Add extra 1 pixel so the click is definitely within the cell instead of maybe on the boundary.
        let x = self.cell_positions[cord.0].0 + 1;
        let y = self.cell_positions[cord.1 * self.board_cell_width as usize].1 + 1;
        enigo.mouse_move_to(x, y);
        enigo.mouse_down(enigo::MouseButton::Left);
        enigo.mouse_up(enigo::MouseButton::Left);
    }

    fn cell_cord_to_offset(&self, cord: CellCord) -> usize {
        let x_offset = cord.0;
        let y_offset = self.board_cell_width as usize * (cord.1);
        let total_offset = x_offset + y_offset;
        return total_offset;
    }

    // Convert cell based cordinate to actual pixel position.
    fn cell_cord_to_pos(&self, cord: CellCord) -> Point {
        let offset = self.cell_cord_to_offset(cord);
        return self.cell_positions[offset];
    }

    fn state_at_cord(&self, cord: CellCord) -> CellKind {
        let offset = self.cell_cord_to_offset(cord);
        return self.state[offset];
    }

    /// Don't use this. It is only public to do a benchmark.
    pub fn identify_cell_benchmark_pub_func(&self) {
        self.identify_cell(CellCord(4, 3)).unwrap();
    }

    fn identify_cell(&self, cord: CellCord) -> Result<CellKind, GameError> {
        // Only unchecked cells can update so don't bother checking if it wasn't an unchecked cell last time it was updated.
        let mut temp: CellKind = self.state_at_cord(cord);
        if temp != CellKind::Unexplored {
            return Ok(temp);
        }
        // This position in the raw screen position. Not the board pixel position.
        // Because this pixel value is being used for cropping of the board it must be relative to the top left of the board.
        let pos = self.cell_cord_to_pos(cord);
        // Have to adjust pixel position because the screenshot is not of the whole screen, but rather just the board.

        // Get section of the board that must be identified as a particular cell kind.
        let sectioned_board_image = imageops::crop_imm(
            &self.board_screenshot,
            pos.0 as u32 - self.top_left.0 as u32,
            pos.1 as u32 - self.top_left.1 as u32,
            self.individual_cell_width,
            self.individual_cell_height,
        );

        let section_of_board = sectioned_board_image.to_image().clone();
        // image must be dynamic image for exact_image_match function
        let section_of_board = DynamicImage::from(section_of_board);

        // Return the first cell type that matches the tile at the given cordinate.
        for (cell_image, cell_kind) in self.cell_images.iter().zip(all::<CellKind>()) {
            // DEBUG
            // save_image("test/Im2.png", DynamicImage::from(self.board_screenshot.clone()));

            // DEBUG
            // save_image("test/Im.png", DynamicImage::from(section_of_board.clone()));

            // See if it is a match.
            let is_match = exact_image_match(
                &DynamicImage::from(cell_image.clone()),
                &section_of_board,
            );

            // If it is a match return that match and stop searching.
            if is_match {
                temp = cell_kind;
                return Ok(temp);
            }
        }

        // If all cell images were matched and none worked.
        println!("UNIDENTIFIED CELL at cord: {:#?}", cord);

        // If the program can't identify the cell then it shouldn't keep trying to play the game.
        self.save_state_info();
        Err(GameError("Can't identify the cell."))
    }

    fn state_by_cell_cord(&mut self, cord: CellCord) -> &mut CellKind {
        let offset = self.cell_cord_to_offset(cord);
        return &mut self.state[offset];
    }

    fn update_state(&mut self, cord: CellCord) {
        // Only unexplored cells can update so don't bother checking if it wasn't an unexplored cell last time it was updated.
        let cell_state_record = *(self.state_by_cell_cord(cord));
        if cell_state_record != CellKind::Unexplored {
            return;
        }

        // TODO Fix unwrap
        let cell = Cell {
            cord,
            kind: self.identify_cell(cord).unwrap(),
        };
        // If cell state is different from recorded for that cell.
        if cell.kind != cell_state_record {
            // Then update state record for that cell.
            *(self.state_by_cell_cord(cell.cord)) = cell.kind;
            // If the cell is fully explored then check it's neighbors because it is not part of the frontier and neighbors may have also updated.
            if cell.kind == CellKind::Explored {
                // Update state of its neighbors.
                for neighbor in cell.neighbors(1, self.board_cell_width, self.board_cell_height) {
                    self.update_state(neighbor)
                }
            }
            // If it is a number and not a fully explored cell.
            else if matches!(
                cell.kind,
                CellKind::One
                    | CellKind::Two
                    | CellKind::Three
                    | CellKind::Four
                    | CellKind::Five
                    | CellKind::Six
                    | CellKind::Seven
                    | CellKind::Eight
            ) {
                // Add cell to frontier
                self.frontier.push(cell)
            }
        }
    }

    pub fn reveal(&mut self, cord: CellCord) {
        self.click(cord);
        let mut temp_cell_kind = CellKind::Unexplored;
        let mut i = 0;
        while temp_cell_kind == CellKind::Unexplored
        // This will keep looking for a change until one occurs.
        {
            if i >= TIMEOUTS_ATTEMPTS_NUM
            // If it must wait this many times give up because something is wrong.
            {
                panic!("Board won't update. Game Loss?");
            }
            self.get_board_screenshot_from_screen();
            // TODO replace below unwrap with handling errors by saving the unknown image and quitting.
            temp_cell_kind = self.identify_cell(cord).unwrap();
            // If the clicked cell looks like it is an unnumbered and explored cell then check that it's neighbors aren't unexplored.
            // If they are any are unexplored then the clicked cell is not finished loading and it only looks this way because that's how the program is displaying it temporarily while it loads.
            if temp_cell_kind == CellKind::Explored {
                // Make a Cell Object for the clicked cell.
                let clicked_cell = Cell {
                    cord,
                    kind: CellKind::Explored,
                };
                // Iterate through the neighbors of the clicked cell.
                'neighbor_loop: for neighbor in
                    clicked_cell.neighbors(1, self.board_cell_width, self.board_cell_height)
                {
                    // TODO replace unwrap with handling of error by saving unknown image.
                    if self.identify_cell(neighbor).unwrap() == CellKind::Unexplored {
                        // If any neighbors are unexplored then this cell can't be a blank explored
                        temp_cell_kind = CellKind::Unexplored; // set a so while loop will take a new screenshot;
                        break 'neighbor_loop;
                    }
                }
                i += 1;
            }
        }
        self.update_state(cord);
    }

    pub fn solve(&mut self) {
        // Reveal initial tile.
        let center = CellCord((self.board_cell_width / 2) as usize, (self.board_cell_height / 2) as usize);
        self.reveal(center);
    }

    /// Saves information about this Game to file for potential debugging purposes.
    pub fn save_state_info(&self) {
        // save saved game state as picture
        // TODO reimplement in rust // game.showGameSavedState().save("FinalGameState.png");

        // Save game state as csv.
        // Write create truncate can be set with `let log_file = File::create(&log_file_name).unwrap();` or as below.
        // Write state to file after formatting nicely.
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open("test/FinalGameState.csv")
            .unwrap();

        for (i, cell) in self.state.iter().enumerate() {
            let symbol_to_write = match cell {
                CellKind::One => '1',
                CellKind::Two => '2',
                CellKind::Three => '3',
                CellKind::Four => '4',
                CellKind::Five => '5',
                CellKind::Six => '6',
                CellKind::Seven => '7',
                CellKind::Eight => '8',
                CellKind::Flag => 'F',
                CellKind::Unexplored => 'U',
                CellKind::Explored => 'E',
            };
            if let Err(e) = write!(file, "{symbol_to_write} ") {
                eprintln!("Couldn't write to file: {}", e);
            }
            if (i + 1) % self.board_cell_width as usize == 0 {
                if let Err(e) = write!(file, "\n") {
                    eprintln!("Couldn't write to file: {}", e);
                }
            }
        }
    }
}

pub fn read_image(path: &str) -> DynamicImage {
    return io::Reader::open(path)
        .expect("Couldn't read image.")
        .decode()
        .expect("Unsupported Type");
}

pub fn save_image(path: &str, image: DynamicImage) {
    save_rgb_vector(
        path,
        image.clone().into_rgb8().to_vec(),
        image.width(),
        image.height(),
    );
}

struct Cell {
    cord: CellCord,
    kind: CellKind,
}

impl Cell {
    // Returns cords of all neighbors that exist.
    fn neighbors(
        &self,
        radius: usize,
        board_cell_width: u32,
        board_cell_height: u32,
    ) -> Vec<CellCord> {
        let mut neighbors = Vec::new();
        // Goes from left to right and from top to bottom generating neighbor cords.
        // Each radius increases number of cells in each dimension by 2 starting with 1 cell at radius = 0
        for j in 0..2 * radius + 1 {
            for i in 0..2 * radius + 1 {
                // TODO check that usize doesn't overflow here for negative cords.
                let x: i64 = self.cord.0 as i64 - radius as i64 + i as i64;
                let y: i64 = self.cord.1 as i64 - radius as i64 + j as i64;
                // Don't make neighbors with negative cords.
                if x < 0 || y < 0 {
                    continue;
                }
                // If neither is negative can safely convert to unsigned.
                let x = x as usize;
                let y = y as usize;

                // Don't make neighbors with cords beyond the bounds of the board.
                if x > board_cell_width as usize - 1 || y > board_cell_height as usize - 1 {
                    continue;
                }

                // Don't add self to neighbor list.
                if x == self.cord.0 && y == self.cord.1 {
                    continue;
                }

                neighbors.push(CellCord(x, y));
            }
        }

        return neighbors;
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
        let super_image = read_image("test_in/subimage_search/sub_board.png");
        let sub_image = read_image("test_in/subimage_search/cell.png");
        let all_positions = locate_all(&super_image, &sub_image);
        assert!(all_positions.iter().any(|point| *point == Point(10, 48)));
        assert!(all_positions.iter().any(|point| *point == Point(26, 48)));
        assert_eq!(all_positions.len(), 2);
    }

    #[test]
    fn test_board_sub_image_search() {
        let super_image = read_image("test_in/subimage_search/board.png");
        let sub_image = read_image("test_in/subimage_search/cell.png");
        let all_positions = locate_all(&super_image, &sub_image);
        assert!(all_positions.iter().any(|point| *point == Point(10, 48)));
        assert!(all_positions.iter().any(|point| *point == Point(26, 48)));
        assert_eq!(all_positions.len(), 16 * 30);
    }

    /// This test requires manual confirmation.
    /// This test requires manually opening a copy of the game first.
    #[ignore]
    #[test]
    fn open_game_first_click_1_1() {
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

    /// This test requires manually opening a copy of the game first.
    /// This test requires partial manual confirmation.
    #[ignore]
    #[test]
    fn open_game_first_reveal() {
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
        let place_to_click = CellCord(7, 7);
        game.get_board_screenshot_from_screen();
        game.reveal(place_to_click);

        game.save_state_info();
        // First move always gives a fully unexplored cell.
        assert_eq!(game.state_at_cord(place_to_click), CellKind::Explored);
    }

    /// This test requires manual confirmation.
    #[ignore]
    #[test]
    fn manual_check_get_board_screenshot_from_screen_test() {
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
        game.get_board_screenshot_from_screen();
        save_rgb_vector(
            "board_screenshot_test.png",
            game.board_screenshot.to_vec(),
            game.board_screenshot.width(),
            game.board_screenshot.height(),
        );
    }

    #[test]
    fn identify_cell_test() {
        let game = Game::build(
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
            read_image("test_in/subimage_search/board.png").to_rgb8(),
            setup_capturer(0),
        );

        assert_eq!(
            game.identify_cell(CellCord(3, 3)).unwrap(),
            CellKind::Unexplored
        );
        assert_eq!(
            game.identify_cell(CellCord(9, 9)).unwrap(),
            CellKind::Unexplored
        );
    }
}
