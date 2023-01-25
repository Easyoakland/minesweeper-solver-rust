use captrs::{Bgr8, Capturer};
use image::{imageops, io, DynamicImage, GenericImageView, /* ImageBuffer,  Rgb,*/ RgbImage};
/* use imageproc::rgb_image; */
use enigo::{Enigo, MouseControllable};
use enum_iterator::{all, Sequence};
use itertools::Itertools;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
    fs::OpenOptions,
    hash::{Hash, Hasher},
    io::prelude::*,
    ops::{Add, Sub},
};

// TODO replace "as usize" with .try_into().unwrap()
// TODO change usize operations with checked_sub or checked_add etc..

const TIMEOUTS_ATTEMPTS_NUM: u8 = 10;
const MAX_COMBINATIONS: u64 = 2000000;

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
                // The capacity of rgb_vec should be the 3 times (rgb) for each rgba pixel since r g and b are separately pulled out but frame has them as a single element.
                let mut rgb_vec = Vec::with_capacity(3 * frame.capacity());
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

/* /// Returns a vector of concatenated RGB values corresponding to a captured frame.
/// Works on a given vector so allocation of that vector can be done once for the program.
fn capture_rgb_frame_in_place(capturer: &mut Capturer, rgb_vec: &mut Vec<u8>) {
    loop {
        let temp = capturer.capture_frame();
        match temp {
            Ok(frame) => {
                // Clear the vector for allocation.
                rgb_vec.clear();
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
            }
            Err(_) => continue,
        }
    }
} */

/// Captures and returns a screenshot as RgbImage.
pub fn capture_image_frame(capturer: &mut Capturer) -> RgbImage {
    return RgbImage::from_raw(
        capturer.geometry().0,
        capturer.geometry().1,
        capture_rgb_frame(capturer),
    )
    .expect("Frame was unable to be captured and converted to RGB.");
}

/* /// Captures and returns a screenshot as RgbImage.
pub fn capture_image_frame_in_place(capturer: &mut Capturer, rgb_vec: &mut Vec<u8>) {
    capture_rgb_frame_in_place(capturer, rgb_vec);
} */

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

// Takes a vector of CellGroup and separates any subsets from supersets.
fn remove_complete_cell_group_overlaps(
    mut cell_group_vec: Vec<CellGroup>,
) -> (bool, Vec<CellGroup>) {
    let mut did_something = false;
    for i in 0..cell_group_vec.len() {
        // Temporary set to see if it is a subset.
        let set1 = cell_group_vec[i].offsets.clone();
        for j in 0..cell_group_vec.len() {
            // Temporary set to see if it is a superset.
            let set2 = &mut cell_group_vec[j].offsets;
            // If it is a different set and it is a subset.
            if (i != j) && set1.is_subset(&set2) {
                // Remove items that are shared from the superset.
                // Below code should be equivalent to next line. Not using next line because it is an unstable library feature.
                // let set2: HashSet<usize> = set2.drain_filter(|elem| set1.contains(elem)).collect();
                let set3: HashSet<usize> = set2.difference(&set1).map(|x| *x).collect();
                *set2 = set3;

                did_something = true;

                // Decrease the mine count of the superset by the subset's minecount.
                cell_group_vec[j].mine_num =
                    cell_group_vec[j].mine_num - cell_group_vec[i].mine_num;
            }
        }
    }
    return (did_something, cell_group_vec);
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

impl CellKind {
    /// Returns a number corresponding to the type of cell.
    /// # Examples
    /// `assert_eq!(CellKind::One.value(), 1);`
    ///
    /// `assert_eq!(CellKind::Eight.value(), 8);`
    fn value(&self) -> Option<u32> {
        return match *self {
            CellKind::One => Some(1),
            CellKind::Two => Some(2),
            CellKind::Three => Some(3),
            CellKind::Four => Some(4),
            CellKind::Five => Some(5),
            CellKind::Six => Some(6),
            CellKind::Seven => Some(7),
            CellKind::Eight => Some(8),
            CellKind::Flag => None,
            CellKind::Unexplored => None,
            CellKind::Explored => None,
        };
    }
    fn value_to_cell_kind(value: u32) -> Result<CellKind, Box<dyn Error>> {
        return match value {
            1 => Ok(CellKind::One),
            2 => Ok(CellKind::Two),
            3 => Ok(CellKind::Three),
            4 => Ok(CellKind::Four),
            5 => Ok(CellKind::Five),
            6 => Ok(CellKind::Six),
            7 => Ok(CellKind::Seven),
            8 => Ok(CellKind::Eight),
            0 => Ok(CellKind::Explored),
            x => Err(Box::from(format!(
                "Invalid input. Can't convert {} to a CellKind.",
                x
            ))),
        };
    }
}

/// Outputs an unordered vector. The vector is the result of merging all sets that share any item in common.
/// This transitively applies until all elements that are somehow linked through these sets are in a single set.
/// The resulting vector contains pairwise disjoint sets.
/// # Examples
/// ```
/// use minesweeper_solver_in_rust::merge_overlapping_sets;
/// use std::collections::HashSet;
///
/// assert_eq!(
/// merge_overlapping_sets(Vec::from([HashSet::from([1,2,3]), HashSet::from([1,2]), HashSet::from([4])])),
/// vec![HashSet::from([1, 2, 3]), HashSet::from([4])]);
///
/// assert_eq!(
/// merge_overlapping_sets(Vec::from([HashSet::from([1,2,3,4]), HashSet::from([1,2]), HashSet::from([4])])),
/// vec![HashSet::from([1, 3, 2, 4])]);
///
/// assert_eq!(
/// merge_overlapping_sets(Vec::from([HashSet::from([1,2,3,4]), HashSet::from([1,2]), HashSet::from([4]), HashSet::from([5,6]), HashSet::from([7])])),
/// vec![HashSet::from([3, 1, 4, 2]), HashSet::from([5, 6]), HashSet::from([7])]);
///
/// assert_eq!(
/// merge_overlapping_sets(Vec::from([HashSet::from([1,2,3,4]), HashSet::from([1,2]), HashSet::from([4]), HashSet::from([5,6]), HashSet::from([4, 7])])),
/// vec![HashSet::from([3, 1, 4, 2, 7]), HashSet::from([5, 6])]);
///
/// assert_eq!(
/// merge_overlapping_sets(Vec::from([HashSet::from(['a','b','c','d']), HashSet::from(['a','g']), HashSet::from(['e']), HashSet::from(['f','h']), HashSet::from(['k', 'a']), HashSet::from(['k', 'z'])])),
/// vec![HashSet::from(['z', 'd', 'b', 'c', 'a', 'k', 'g']), HashSet::from(['e']), HashSet::from(['h', 'f'])]);
/// ```
pub fn merge_overlapping_sets<T>(sets: Vec<HashSet<T>>) -> Vec<HashSet<T>>
where
    T: Eq + Hash + Clone,
{
    let mut merged_sets: Vec<HashSet<T>> = Vec::new();
    for s in sets {
        let mut is_overlapping = false;

        for t in &mut merged_sets {
            let intersection: HashSet<T> = s.intersection(t).cloned().collect();
            if !intersection.is_empty() {
                let union: HashSet<T> = s.union(t).cloned().collect();
                *t = union;
                is_overlapping = true;
                break;
            }
        }

        if !is_overlapping {
            merged_sets.push(s);
        }
    }
    let mut result: Vec<HashSet<T>> = Vec::new();
    for set in merged_sets {
        let mut flag = true;
        for s in &result {
            if s.is_subset(&set) {
                flag = false;
                break;
            }
        }
        if flag {
            result.push(set);
        }
    }
    result
}

fn merge_overlapping_groups<'a, T>(groups: &'a Vec<CellGroup>) -> Vec<HashSet<&'a CellGroup>>
where
    T: Eq + Hash + Clone,
    HashSet<T>: FromIterator<usize>,
{
    let mut merged_groups: Vec<HashSet<&CellGroup>> = Vec::new();
    for group in groups {
        let mut is_overlapping = false;

        for t in &mut merged_groups {
            let intersection: HashSet<T> = group
                .offsets
                .intersection(&t.iter().next().unwrap().offsets)
                .cloned()
                .collect();
            if !intersection.is_empty() {
                t.insert(&group);
                is_overlapping = true;
                break;
            }
        }

        if !is_overlapping {
            let mut set = HashSet::new();
            set.insert(group);
            merged_groups.push(set);
        }
    }
    merged_groups
}

/// Simple factorial calculation for positive integers.
/// # Examples
/// ```
/// use minesweeper_solver_in_rust::factorial;
/// assert_eq!(factorial(3), 6);
/// ```
pub fn factorial(n: u64) -> u64 {
    (2..(n + 1)).product()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CellGroup {
    offsets: HashSet<usize>,
    mine_num: u32,
}

impl Hash for CellGroup {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for offset in self.offsets.iter() {
            offset.hash(state);
        }
    }
}

fn neighbors_of_cord(
    cord: CellCord,
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
            let x: i64 = cord.0 as i64 - radius as i64 + i as i64;
            let y: i64 = cord.1 as i64 - radius as i64 + j as i64;
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
            if x == cord.0 && y == cord.1 {
                continue;
            }

            neighbors.push(CellCord(x, y));
        }
    }

    return neighbors;
}

fn cell_cord_to_offset(board_cell_width: u32, cord: CellCord) -> usize {
    let x_offset = cord.0;
    let y_offset = board_cell_width as usize * (cord.1);
    let total_offset = x_offset + y_offset;
    return total_offset;
}

fn offset_to_cell_cord(board_cell_width: u32, mut offset: usize) -> CellCord {
    let mut cnt = 0;
    while offset >= board_cell_width as usize {
        offset = offset - board_cell_width as usize;
        cnt += 1
    }
    let y = cnt;
    let x = offset;
    // y = int(Offset/self._width)
    // x = int(Offset-y*self._width)
    return CellCord(x, y);
}

/* fn neighbors_of_offset(
    mut offset: usize,
    radius: usize,
    board_cell_width: u32,
    board_cell_height: u32,
) -> Vec<CellCord> {
    let cord = {
        let mut cnt = 0;
        while offset >= board_cell_width as usize {
            offset = offset - board_cell_width as usize;
            cnt += 1
        }
        let y = cnt;
        let x = offset;
        // y = int(Offset/self._width)
        // x = int(Offset-y*self._width)
        CellCord(x, y)
    };
    return neighbors_of_cord(cord, radius, board_cell_width, board_cell_height);
} */

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
        return neighbors_of_cord(self.cord, radius, board_cell_width, board_cell_height);
    }
}

#[derive(Debug)]
pub struct Simulation {
    state: Vec<bool>,
    board_cell_width: u32,
    board_cell_height: u32,
}

impl Simulation {
    pub fn new(width: u32, height: u32, mine_num: u32) -> Simulation {
        let mut state = Vec::with_capacity(width as usize * height as usize);

        // Generate random unique indexes until there are enough to represent each mine.
        let mut mine_pos: Vec<u32> = vec![];
        while mine_pos.len() != mine_num as usize {
            let random_num = rand::random::<u32>() % (width * height);
            if !mine_pos.contains(&random_num) {
                mine_pos.push(random_num);
            }
        }

        // Make a state where each position is not a mine unless its index matches one of the mine positions.
        for h in 0..height {
            for w in 0..width {
                state.push(mine_pos.contains(&(h * width + w)));
            }
        }
        return Simulation {
            state,
            board_cell_width: width,
            board_cell_height: height,
        };
    }

    pub fn is_mine(&self, index: usize) -> bool {
        return self.state[index];
    }

    /// Returns none if the item is a mine and otherwise returns how many mines are in the neighborhood.
    pub fn value(&self, index: usize) -> Option<u32> {
        if self.is_mine(index) {
            return None;
        }
        let mut surrounding_mines = 0;
        for neighbor in neighbors_of_cord(
            offset_to_cell_cord(self.board_cell_width, index),
            1,
            self.board_cell_width,
            self.board_cell_height,
        ) {
            if self.state[cell_cord_to_offset(self.board_cell_width, neighbor)] {
                surrounding_mines += 1
            }
        }
        return Some(surrounding_mines);
    }
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
    // bottom_right: Point,
    individual_cell_width: u32,
    individual_cell_height: u32,
    board_screenshot: RgbImage,
    capturer: Capturer,
    frontier: Vec<Cell>,
    cell_groups: Vec<CellGroup>,
    simulation: Option<Simulation>,
    /* board_screenshot_vec: Vec<u8>, */
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

        // Setting the bottom right requires finding the top left of the last cell and shifting by the cell's width and height to get to the bottom right.
        // let bottom_right = cell_positions
        //     .last()
        //     .expect("Will only fail if there is no game grid with cells");
        // let offset_from_top_left_corner_to_bottom_right_corner_of_cell =
        //     Point(individual_cell_width as i32, individual_cell_height as i32);
        // Update bottom_right by offset to go from top left to bottom right of last cell.
        // let bottom_right =
        //     *bottom_right + offset_from_top_left_corner_to_bottom_right_corner_of_cell;

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
            // bottom_right,
            individual_cell_width,
            individual_cell_height,
            capturer,
            board_screenshot: screenshot, // Initially not cropped to just board. Will be when it is set using the correct method.
            frontier: Vec::new(),
            cell_groups: Vec::new(),
            simulation: None,
            /* board_screenshot_vec: Vec::with_capacity((screenshot.width()*screenshot.height()) as usize) */
        };
    }

    pub fn new(cell_files: [&str; CELL_VARIANT_COUNT], id: usize) -> Game {
        let mut capturer = setup_capturer(id);

        // Finds the initial positions of the cells in the game grid.
        let screenshot = capture_image_frame(&mut capturer);
        return Game::build(cell_files, screenshot, capturer);
    }

    pub fn new_for_simulation() -> Game {
        let board_cell_width:u32 = 30;
        let board_cell_height:u32 = 16;
        let mine_num = 99;
        let capturer = setup_capturer(0);
        // Finds the initial positions of the cells in the game grid.
        let board_screenshot = RgbImage::from_vec(0, 0, vec![]).unwrap();
        // Initializes state as all unexplored.
        let state = vec![CellKind::Unexplored; (board_cell_height * board_cell_width)as usize];
        let simulation = Some(Simulation::new(board_cell_width, board_cell_height, mine_num));
        Game {
            simulation,
            board_cell_height,
            board_cell_width,
            state,
            frontier: Vec::new(),
            cell_groups: Vec::new(),
            // Below are all set to whatever (0 mostly) because they don't impact the simulation.
            // They are primarily parameters of the screen and image of the board.
            cell_positions: Vec::new(),
            cell_images: Vec::new(),
            board_px_height: 0,
            board_px_width: 0,
            capturer,
            board_screenshot,
            top_left: Point(0,0),
            individual_cell_width: 0,
            individual_cell_height: 0,
        }
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

    /*     /// Sets the board screenshot of Game to just the board of tiles. Crops out extra stuff.
    fn get_board_screenshot_from_screen_in_place(&mut self) {
        capture_image_frame_in_place(&mut self.capturer, &mut self.board_screenshot_vec);
        let (width,height) = self.capturer.geometry();
        let screenshot = RgbImage::from_vec(width, height, self.board_screenshot_vec).expect("Failed to convert vector to RGBImage");
        self.board_screenshot = image::imageops::crop_imm(
            &screenshot,
            self.top_left.0 as u32,
            self.top_left.1 as u32,
            self.board_px_width,
            self.board_px_height,
        )
        .to_image();
    } */

    /// Left clicks indicated cord
    fn click_left_cell_cord(&self, cord: CellCord) {
        let mut enigo = Enigo::new();
        // Add extra 1 pixel so the click is definitely within the cell instead of maybe on the boundary.
        let x = self.cell_positions[cord.0].0 + 1;
        let y = self.cell_positions[cord.1 * self.board_cell_width as usize].1 + 1;
        enigo.mouse_move_to(x, y);
        enigo.mouse_down(enigo::MouseButton::Left);
        enigo.mouse_up(enigo::MouseButton::Left);
    }

    /// Right click indicated cord
    fn click_right_cell_cord(&self, cord: CellCord) {
        let mut enigo = Enigo::new();
        // Add extra 1 pixel so the click is definitely within the cell instead of maybe on the boundary.
        let x = self.cell_positions[cord.0].0 + 1;
        let y = self.cell_positions[cord.1 * self.board_cell_width as usize].1 + 1;
        enigo.mouse_move_to(x, y);
        enigo.mouse_down(enigo::MouseButton::Right);
        enigo.mouse_up(enigo::MouseButton::Right);
    }

    fn cell_cord_to_offset(&self, cord: CellCord) -> usize {
        cell_cord_to_offset(self.board_cell_width, cord)
    }

    fn offset_to_cell_cord(&self, offset: usize) -> CellCord {
        offset_to_cell_cord(self.board_cell_width, offset)
    }

    // Convert cell based cordinate to actual pixel position.
    fn cell_cord_to_pos(&self, cord: CellCord) -> Point {
        let offset = self.cell_cord_to_offset(cord);
        return self.cell_positions[offset];
    }

    fn state_at_cord(&mut self, cord: CellCord) -> &mut CellKind {
        let offset = self.cell_cord_to_offset(cord);
        return &mut self.state[offset];
    }

    fn state_at_cord_imm(&self, cord: CellCord) -> &CellKind {
        let offset = self.cell_cord_to_offset(cord);
        return &self.state[offset];
    }

    /// Don't use this. It is only public to do a benchmark.
    pub fn identify_cell_benchmark_pub_func(&mut self) {
        self.identify_cell(CellCord(4, 3)).unwrap();
    }

    fn identify_cell(&mut self, cord: CellCord) -> Result<CellKind, GameError> {
        // Only unchecked cells can update so don't bother checking if it wasn't an unchecked cell last time it was updated.
        let mut temp: CellKind = *self.state_at_cord(cord);
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
            // See if it is a match.
            let is_match =
                exact_image_match(&DynamicImage::from(cell_image.clone()), &section_of_board);

            // If it is a match return that match and stop searching.
            if is_match {
                temp = cell_kind;
                return Ok(temp);
            }
        }

        // If all cell images were matched and none worked.
        println!(
            "UNIDENTIFIED CELL at cord: {:?}(offset: {:?})",
            cord,
            self.cell_cord_to_offset(cord)
        );
        save_image("cell_images/Unidentified.png", section_of_board);
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
        self.click_left_cell_cord(cord);
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

    fn identify_cell_simulation(&self, cord: CellCord) -> Result<CellKind, Box<dyn Error>> {
        CellKind::value_to_cell_kind(
            self.simulation
                .as_ref()
                .expect("Simulation doesn't exist but function requires it does.")
                .value(cell_cord_to_offset(self.board_cell_width, cord))
                .expect(&format!("Revealed Mine at {cord:?}.").to_string()),
        )
    }

    fn update_state_simulation(&mut self, cord: CellCord) {
        // Only unexplored cells can update so don't bother checking if it wasn't an unexplored cell last time it was updated.
        let cell_state_record = *(self.state_by_cell_cord(cord));
        if cell_state_record != CellKind::Unexplored {
            return;
        }

        // TODO Fix unwrap
        let cell = Cell {
            cord,
            kind: self.identify_cell_simulation(cord).unwrap(),
        };
        // If cell state is different from recorded for that cell.
        if cell.kind != cell_state_record {
            // Then update state record for that cell.
            *(self.state_by_cell_cord(cell.cord)) = cell.kind;
            // If the cell is fully explored then check it's neighbors because it is not part of the frontier and neighbors may have also updated.
            if cell.kind == CellKind::Explored {
                // Update state of its neighbors.
                for neighbor in cell.neighbors(1, self.board_cell_width, self.board_cell_height) {
                    self.update_state_simulation(neighbor)
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

    fn reveal_simulation(&mut self, cord: CellCord) {
        // TODO replace below unwrap. It should somehow indicate that a mine was revealed so game loss.
        self.identify_cell_simulation(cord).unwrap();
        self.update_state_simulation(cord);
    }

    /// Flag cell at cord then update cell state at location to flag.
    fn flag(&mut self, cord: CellCord) {
        // Don't do anything if the cell isn't flaggable.
        if *self.state_at_cord_imm(cord) != CellKind::Unexplored {
            println!("Tried flagging a non flaggable at: {:#?}", cord);
            return;
        }
        // Since the cell is flaggable flag it...
        self.click_right_cell_cord(cord);
        // ...and update the internal state of that cell to match.
        *self.state_at_cord(cord) = CellKind::Flag;
        return;
    }

    /// Flag cell at cord then update cell state at location to flag.
    fn flag_simulation(&mut self, cord: CellCord) {
        // Don't do anything if the cell isn't flaggable.
        if *self.state_at_cord_imm(cord) != CellKind::Unexplored {
            println!("Tried flagging a non flaggable at: {:#?}", cord);
            return;
        }
        // Update the internal state of that cell to match.
        *self.state_at_cord(cord) = CellKind::Flag;
        return;
    }

    /// Rule 1 implemented with sets. If the amount of mines in a set is the same as the amount of cells in a set, they are all mines.
    /// Returns a bool indicating whether the rule did something.
    fn cell_group_rule_1(&mut self, cell_group: &CellGroup, simulate: bool) -> bool {
        // If the number of mines in the set is the same as the size of the set.
        if cell_group.mine_num as usize == cell_group.offsets.len() {
            // Flag all cells in set.
            for offset in cell_group.offsets.iter() {
                if !simulate {
                    self.flag(self.offset_to_cell_cord(*offset))
                } else if simulate {
                    self.flag_simulation(self.offset_to_cell_cord(*offset))
                }
            }
            // Rule activated.
            return true;
        } else {
            // Rule didn't activate.
            return false;
        }
    }

    fn cell_group_rule_2(&mut self, cell_group: &CellGroup, simulate: bool) -> bool {
        // If set of cells has no mine. (mine_num is 0 from previous if)
        if cell_group.mine_num == 0 {
            // The reveals at the end might affect cells that have yet to be changed.
            for offset in cell_group.offsets.iter() {
                // If a previous iteration of this loop didn't already reveal that cell, then reveal that cell.
                if self.state[*offset] == CellKind::Unexplored {
                    if !simulate {
                        self.reveal(self.offset_to_cell_cord(*offset));
                    } else if simulate {
                        self.reveal_simulation(self.offset_to_cell_cord(*offset));
                    }
                }
            }
            return true; // Rule activated.
        } else {
            return false; // Didn't activate.
        }
    }

    // Generates a set for a given cell.
    fn generate_cell_group(&self, cell: Cell) -> Option<CellGroup> {
        // Make a set for all locations given as an offset.
        let mut offsets = HashSet::new();
        let mut flag_cnt = 0;
        // For every neighbor of the given cell.
        for neighbor in cell.neighbors(1, self.board_cell_width, self.board_cell_height) {
            // If the neighbor is unexplored.
            let temp_state = *self.state_at_cord_imm(neighbor);
            if temp_state == CellKind::Unexplored {
                // Add the offset to the set.
                offsets.insert(self.cell_cord_to_offset(neighbor));
            }
            // If the neighbor is a flag add to the count of surrounding flags.
            else if temp_state == CellKind::Flag {
                flag_cnt += 1
            }
        }
        // If set is empty don't return anything because there is no valid CellGroup
        if offsets.len() == 0 {
            return None;
        }

        // Set mine num based on the cell's number.
        if let Some(cell_value) = cell.kind.value() {
            // The amount of mines in the CellGroup is the amount there are around the cell minus how many have already been identified
            let mine_num = cell_value - flag_cnt;
            return Some(CellGroup { offsets, mine_num });
        }
        // If the cell doesn't have a number then return nothing because non-numbered cells don't have associated CellGroup.
        else {
            None
        }
    }

    fn process_frontier(&mut self, simulate: bool) {
        while self.frontier.len() > 0 {
            let current_cell = self
                .frontier
                .pop()
                .expect("Already checked frontier length > 0.");
            let cell_group = self.generate_cell_group(current_cell);

            if let Some(cell_group) = cell_group {
                // If rule 1 was able to do something currentCell.
                // Then the rest of the loop is unnecessary.
                if self.cell_group_rule_1(&cell_group, simulate) {
                }
                // If rule 2 was able to do something to the currentCell.
                // Then the rest of the loop is unnecessary.
                else if self.cell_group_rule_2(&cell_group, simulate) {
                }
                // If neither rule could do something to the currentCell then add it to a list to apply more advanced techniques to later.
                // Then add it to the list of cell_group.
                else {
                    self.cell_groups.push(cell_group);
                }
            }
        }
    }

    // Uses deterministic methods to solve the game.
    // TODO make processing frontier and CellGroup two separate functions.
    fn deterministic_solve(&mut self, simulate: bool) {
        // Makes outermost loop always execute at least once.
        let mut do_while_flag = true;
        // Loops through frontier and self.cell_groups.
        // Continues looping until inner loop indicates self.cell_groups can't be processed anymore and outer loop indicates the frontier is empty.
        while do_while_flag || self.frontier.len() > 0 {
            do_while_flag = false;
            while self.frontier.len() > 0 {
                self.process_frontier(simulate);
            }
            // Set did_someting to 1 so self.cell_groups is processed at least once.
            let mut did_something = 1;
            while did_something > 0 && self.cell_groups.len() > 0 {
                // Set flag to 0 so it will detect no changes as still being 0.
                did_something = 0;

                // Simplify any CellGroup that can be simplified.
                // Following is a for loop of self.cell_groups.len() but the index is changed in some places in the loop so spots aren't missed.
                let mut i = 0;
                // for i in 0..self.cell_groups.len() {
                while i < self.cell_groups.len() {
                    // TODO split to function START -----------------------------------------------------------------------------------------
                    // Check to see if any cell_group now contain a flag or an already explored cell.
                    for offset in self.cell_groups[i].offsets.clone().into_iter() {
                        // If the cell_group now contain a flag.
                        if self.state[offset] == CellKind::Flag {
                            // Remove the flag from the cell_group
                            // and decrease the amount of mines left.
                            if !self.cell_groups[i].offsets.remove(&offset) {
                                panic!(
                                    "Removed offset: {:?} that wasn't in cell_group: {:?}.",
                                    offset, self.cell_groups[i]
                                )
                            };
                            if self.cell_groups[i].offsets.len()
                                < self.cell_groups[i].mine_num as usize - 1
                            {
                                panic!("There are more mines than places to put them. Removed offset: {:?} as it was a flag in in cell_group: {:?}.", offset, self.cell_groups[i]);
                            }
                            self.cell_groups[i].mine_num -= 1;
                            did_something += 1;
                        }
                        // If the cell_group now contains an not unexplored cell remove that cell as it can't be one of the mines anymore.
                        else if self.state[offset] != CellKind::Unexplored {
                            self.cell_groups[i].offsets.remove(&offset);
                            did_something += 1
                        }
                        // Below shouldn't be true ever and exists to detects errors.
                        if self.cell_groups[i].mine_num as usize > self.cell_groups[i].offsets.len()
                        {
                            self.save_state_info();
                            panic!("ERROR at self.cell_groups[{i}]={:?} has more mines than cells to fill.",self.cell_groups[i]);
                        }
                    }
                    // TODO split to function END -----------------------------------------------------------------------------------------

                    // TODO split to function START -----------------------------------------------------------------------------------------
                    // Check if a logical operation can be done.
                    let cell_groups = self.cell_groups[i].clone();

                    if self.cell_group_rule_1(&cell_groups, simulate) {
                        // Since that cell_group is solved it is no longer needed.
                        self.cell_groups.swap_remove(i);
                        // Decrement loop index so this index is not skipped in next iteration now that a new value is in the index's position.
                        i = usize::saturating_sub(i, 1); // Saturate so a 0 index will terminate on next loop.

                        did_something += 1;
                    } else if self.cell_group_rule_2(&cell_groups, simulate) {
                        // Since that cell_group is solved it is no longer needed.
                        self.cell_groups.swap_remove(i);

                        // Decrement loop index so this index is not skipped in next iteration now that a new value is in the index's position.
                        i = usize::saturating_sub(i, 1); // Saturate so a 0 index will terminate on next loop.
                        did_something += 1;
                    }
                    // Increment loop index
                    i += 1;
                }
                // TODO split to function END -----------------------------------------------------------------------------------------

                // Remove subset-superset overlaps.
                {
                    let overlaps_existed;
                    (overlaps_existed, self.cell_groups) =
                        remove_complete_cell_group_overlaps(self.cell_groups.clone());
                    if overlaps_existed {
                        did_something += 1
                    }
                }
            }
        }
    }

    fn enumerate_all_possible_arrangements(
        &self,
        sub_group_mine_num_lower_limit: usize,
        sub_group_mine_num_upper_limit: usize,
        sub_group_total_offsets_after_overlaps_removed: HashSet<usize>,
        sub_group: HashSet<&CellGroup>,
    ) -> (i32, HashMap<usize, i32>) {
        // DEBUG
        let sub_group_for_debug = sub_group.clone();

        // Iterate through all possible amounts of mines in the subgroup.
        // Calculate odds as the amount of times a mine appeared in a position divided by number of valid positions.
        // Division is done because comparison at the end will include other probabilities with different denominators.
        // Holds how often a mine occurs in each offset position.
        let mut occurrences_of_mine_per_offset: HashMap<_, _> =
            sub_group_total_offsets_after_overlaps_removed
                .iter()
                .map(|offset| (*offset, 0))
                .collect();
        let mut number_of_valid_combinations = 0;
        for sub_group_mine_num in
            0.max(sub_group_mine_num_lower_limit)..(sub_group_mine_num_upper_limit + 1)
        {
            // Count up how many times a mine occurs in each offset position.
            'combinations: for combination in sub_group_total_offsets_after_overlaps_removed
                .iter()
                .combinations(sub_group_mine_num)
            {
                // Verifies that the number of mines in this combination is not invalid for each individual CellGroup
                for cell_group in sub_group.iter() {
                    // Stores how many mines are in a CellGroup for this particular arrangement of mines.
                    let mut individual_cell_group_mine_num_for_specific_combination = 0;
                    // For every offset in CellGroup.
                    for offset in cell_group.offsets.iter() {
                        // If the offset is a mine.
                        if combination.contains(&&offset) {
                            // Count up how many mines are in the CellGroup for this arrangement.
                            individual_cell_group_mine_num_for_specific_combination += 1;
                        }
                    }
                    // If the amount of mines isn't the right amount.
                    // Go to the next combination because this one doesn't work.
                    if individual_cell_group_mine_num_for_specific_combination
                        != cell_group.mine_num
                    {
                        continue 'combinations;
                    }
                }
                // Since the amount of mines is correct.
                // For every offset in this valid combination.
                // Increment the amount of mines at each offset.
                for offset in combination {
                    *(occurrences_of_mine_per_offset.get_mut(offset).unwrap()) += 1;
                }

                // Since the arrangement is valid increment the number of valid arrangements.
                number_of_valid_combinations += 1
            }
        }
        if number_of_valid_combinations == 0 {
            dbg!(sub_group_for_debug);
        }
        return (number_of_valid_combinations, occurrences_of_mine_per_offset);
    }

    fn update_likelihoods_from_enumerated_arrangements(
        &self,
        mut most_likely_positions: Vec<usize>,
        mut most_likelihood: f64,
        mut least_likely_positions: Vec<usize>,
        mut least_likelihood: f64,
        number_of_valid_combinations: i32,
        occurrences_of_mine_per_offset: HashMap<usize, i32>,
    ) -> Option<(Vec<usize>, f64, Vec<usize>, f64)> {
        // If there was a valid combination.
        if number_of_valid_combinations > 0
        // TODO remove next line I don't know why it is here.
        // && occurrences_of_mine_per_offset.values().max().unwrap() > &0
        {
            // Enumerate offsets and chances of those offsets.
            for (offset, occurrence_of_mine_at_offset) in occurrences_of_mine_per_offset.iter() {
                // The chance a mine is somewhere is the amount of combinations a mine occurred in that position divided by how many valid combinations there are total.
                let chance_of_mine_at_position: f64 =
                    *occurrence_of_mine_at_offset as f64 / number_of_valid_combinations as f64;

                if chance_of_mine_at_position > most_likelihood {
                    // If likelyhood of mine is higher than previously recorded.
                    // Update likelyhood
                    // and update position with highest likelyhood.
                    most_likelihood = chance_of_mine_at_position;
                    most_likely_positions = vec![*offset];
                }
                // If the likelyhood is 100% then add it anyway because 100% means theres a mine for sure and it should be flagged regardless.
                else if chance_of_mine_at_position == 1.0 {
                    // update likelyhood and append position of garrunteed mine.
                    most_likelihood = 1.0;
                    most_likely_positions.push(*offset);
                }
                // Same thing but for leastlikelyhood.
                if chance_of_mine_at_position < least_likelihood {
                    least_likelihood = chance_of_mine_at_position;
                    least_likely_positions = vec![*offset];
                }
                // If the chance of a mine is zero then it is guaranteed to not have a mine and should be revealed regardless.
                else if chance_of_mine_at_position == 0.0 {
                    least_likelihood = 0.0;
                    least_likely_positions.push(*offset);
                }
            }
            return Some((
                most_likely_positions,
                most_likelihood,
                least_likely_positions,
                least_likelihood,
            ));
        } else {
            return None;
        };
    }

    /// Make best guess from all possibilities.
    fn probabalistic_guess(&mut self, simulate: bool) -> u32 {
        let mut did_something = 0;

        // Keep track of the most and least likely places for there to be a mine and the likelyhood of each.
        let mut most_likely_positions = Vec::new();
        let mut most_likelihood = -1.0;
        let mut least_likely_positions = Vec::new();
        let mut least_likelihood = 101.0;

        // TODO remove below
        // Get a vec of all the offsets so they can be merged in next step.
        // let mut offset_vec = Vec::with_capacity(self.cell_groups.len());
        // for cell_group in &self.cell_groups {
        //     offset_vec.push(cell_group.offsets.clone());
        // }

        // Find the sub groups of the grid of interconnected cell_groups that are not related or interconnected.
        // Basically partitions board so parts that don't affect each other are handled separately to make the magnitudes of the combinations later on more managable.
        let sub_groups = merge_overlapping_groups(&self.cell_groups);

        // For each independent sub group of cell_groups.
        for sub_group in sub_groups {
            let mut sub_group_total_offsets: Vec<usize> = Vec::new();
            let mut sub_group_mine_num_upper_limit_for_completely_unshared_mines = 0;

            // Put all offsets of corresponding subgroup into a Vec. Also count how many mines exist if there are no duplicates.
            for cell_group in sub_group.iter() {
                sub_group_total_offsets.extend(cell_group.offsets.clone());

                // The upper limit on the number of mines in a subgroup is if all the CellGroup share no mines.
                // This number is the sum total of simply adding each mine_num.
                sub_group_mine_num_upper_limit_for_completely_unshared_mines += cell_group.mine_num;
            }

            // Save how many offsets with overlaps there are.
            // Do this by saving how many offsets exist before (here) and after (later) merging.
            let number_of_subgroup_total_offsets_before_overlaps_removed =
                sub_group_total_offsets.len();

            // Remove overlaps here by converting to a set which by defintion contains no duplicates.
            let sub_group_total_offsets_after_overlaps_removed: HashSet<usize> =
                sub_group_total_offsets.into_iter().collect();
            let number_of_sub_group_total_offsets_after_overlaps_removed =
                sub_group_total_offsets_after_overlaps_removed.len();

            // An upper limit of mines is if every intersection does not have a mine.
            // Another is the number of positions in the sub_group. It can't have more mines than it has positions.
            // Set upperlimit to the smaller upperlimit.
            let sub_group_mine_num_upper_limit =
                number_of_sub_group_total_offsets_after_overlaps_removed
                    .min(sub_group_mine_num_upper_limit_for_completely_unshared_mines as usize);

            // The lower limit on mines is if every intersection had a mine.
            // It is the same as upper limit (no mines at intersections or number of places for a mine to be) minus the number of intersections.
            // This is because each intersection is another place where the mine could have been double counted in the upper limit.
            // Saturating subtraction because lower_limit can't be negative but the subtraction might if there are more overlaps than maximum upper limit.
            let sub_group_mine_num_lower_limit = sub_group_mine_num_upper_limit.saturating_sub(
                number_of_subgroup_total_offsets_before_overlaps_removed
                    - number_of_sub_group_total_offsets_after_overlaps_removed,
            );

            // DEBUG
            // dbg!(sub_group_mine_num_upper_limit);
            // dbg!(number_of_subgroup_total_offsets_before_overlaps_removed);
            // dbg!(number_of_sub_group_total_offsets_after_overlaps_removed);

            // Check that the amount of combinations will not exceed the global variable for the maximum combinations.
            // If it does this will take too long.
            let mut combination_total = 0;
            // From the least to the most possible number of mines.
            for sub_group_mine_num in
                0.max(sub_group_mine_num_lower_limit)..(sub_group_mine_num_upper_limit + 1)
            {
                // Calculate the amount of combinations. Integer division means it might be off by one but that doesn't matter.
                let combination_amount =
                    factorial(sub_group_total_offsets_after_overlaps_removed.len() as u64)
                        / (factorial(
                            sub_group_total_offsets_after_overlaps_removed.len() as u64
                                - sub_group_mine_num as u64,
                        ) * factorial(sub_group_mine_num as u64));
                combination_total += combination_amount;

                // DEBUG
                // println!("{sub_group_total_offsets} with length {sub_group_total_offsets.len()} pick {sub_group_mine_num} is {combination_amount} total combinations")

                // If there are too many combinations then abort calculation.
                if combination_total > MAX_COMBINATIONS {
                    println!("Not computing {combination_total} total combinations. using fast guess instead");
                    // Return that nothing was done.
                    return 0;
                }
            }

            let (number_of_valid_combinations, occurrences_of_mine_per_offset) = self
                .enumerate_all_possible_arrangements(
                    sub_group_mine_num_lower_limit,
                    sub_group_mine_num_upper_limit,
                    sub_group_total_offsets_after_overlaps_removed,
                    sub_group,
                );

            (
                most_likely_positions,
                most_likelihood,
                least_likely_positions,
                least_likelihood,
            ) = match self.update_likelihoods_from_enumerated_arrangements(
                most_likely_positions,
                most_likelihood,
                least_likely_positions,
                least_likelihood,
                number_of_valid_combinations,
                occurrences_of_mine_per_offset,
            ) {
                Some(x) => x,
                None => {
                    self.save_state_info();
                    dbg!(sub_group_mine_num_lower_limit);
                    dbg!(sub_group_mine_num_upper_limit);
                    panic!("There were no valid combinations!")
                }
            };
        }
        // TODO make code below new function.
        // If more certain about where a mine isn't than where one is.
        if most_likelihood <= 1.0 - least_likelihood {
            // Then reveal all spots with lowest odds of mine.
            for least_likely_position in least_likely_positions {
                println!(
                    "Revealing {:?} with odds {:?} of being mine",
                    self.offset_to_cell_cord(least_likely_position),
                    least_likelihood
                );
                if !simulate {
                    self.reveal(self.offset_to_cell_cord(least_likely_position));
                } else if simulate {
                    self.reveal_simulation(self.offset_to_cell_cord(least_likely_position))
                }
            }
            did_something += 1;
        }
        // If more certain about where a mine is than where one isn't.
        else if most_likelihood > 1.0 - least_likelihood {
            // Then flag all spots with lowest odds of mine.
            for most_likely_position in most_likely_positions {
                println!(
                    "Flagging {:?} with odds {:?} of being mine",
                    self.offset_to_cell_cord(most_likely_position),
                    most_likelihood
                );
                if !simulate {
                    self.flag(self.offset_to_cell_cord(most_likely_position));
                } else if simulate {
                    self.flag_simulation(self.offset_to_cell_cord(most_likely_position));
                }
                did_something += 1;
            }
        }
        return did_something;
    }

    pub fn solve(&mut self, initial_guess: CellCord, simulate: bool) {
        if initial_guess.0 > self.board_cell_width as usize {
            panic!("Initial guess is larger than the board width.");
        } else if initial_guess.1 > self.board_cell_height as usize {
            panic!("Initial guess is larger than the board height.");
        }
        if !simulate {
            // Reveal initial tile.
            self.reveal(initial_guess);
        }
        else if simulate{
            self.reveal_simulation(initial_guess);
        }

        let mut did_something = 1;
        while did_something > 0 {
            // Did something is set to 0 so the loop will only continue is something happens to change it.
            did_something = 0;

            self.deterministic_solve(simulate);
            if self.cell_groups.len() > 0 {
                print!("Guess required. ");
                if self.probabalistic_guess(simulate) >= 1 {
                    did_something += 1;
                    continue;
                }
            }
        }
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

    /// Saves information about this Game to file for potential debugging purposes.
    pub fn save_state_info_with_path(&self, path: &str) {
        // save saved game state as picture
        // TODO reimplement in rust // game.showGameSavedState().save("FinalGameState.png");

        // Save game state as csv.
        // Write create truncate can be set with `let log_file = File::create(&log_file_name).unwrap();` or as below.
        // Write state to file after formatting nicely.
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
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
        let super_image = read_image("test_in/subimage_search/board_plus_extra.png");
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

        game.click_left_cell_cord(CellCord(1, 1));
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
        assert_eq!(*game.state_at_cord(place_to_click), CellKind::Explored);
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
        let mut game = Game::build(
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

    #[test]
    fn remove_complete_cell_group_overlaps_test() {
        let cell_group_vec = vec![
            CellGroup {
                mine_num: 3,
                offsets: HashSet::from([1, 2, 3, 4, 5]),
            },
            CellGroup {
                mine_num: 2,
                offsets: HashSet::from([1, 2, 3, 4]),
            },
        ];
        let (result_bool, result_of_no_overlap) =
            remove_complete_cell_group_overlaps(cell_group_vec);
        assert!(result_bool);
        assert!(result_of_no_overlap.contains(&CellGroup {
            offsets: HashSet::from([4, 2, 1, 3]),
            mine_num: 2,
        }));
        assert!(result_of_no_overlap.contains(&CellGroup {
            offsets: HashSet::from([5]),
            mine_num: 1,
        }));
    }

    #[test]
    fn merge_overlapping_groups_test() {
        let cell1 = CellGroup {
            offsets: HashSet::from([1, 2]),
            mine_num: 1,
        };
        let cell2 = CellGroup {
            offsets: HashSet::from([1, 2, 3]),
            mine_num: 2,
        };
        let cell3 = CellGroup {
            offsets: HashSet::from([8, 9, 10]),
            mine_num: 2,
        };
        let groups = vec![cell1.clone(), cell2.clone(), cell3.clone()];
        let output = merge_overlapping_groups(&groups);
        assert!(&output[0].contains(&cell1));
        assert!(&output[0].contains(&cell2));
        assert!(&output[1].contains(&cell3));
    }

    #[test]
    fn test_simulation_creation() {
        let sim = Simulation::new(4, 5, 6);
        assert_eq!(sim.board_cell_width, 4);
        assert_eq!(sim.board_cell_height, 5);
        assert_eq!(sim.state.iter().filter(|x| **x).count(), 6);
    }

    #[test]
    fn simulate_solve() {
        let mut game = Game::new_for_simulation();
        game.solve(CellCord(0,0), true);
        dbg!(game.state);

    }
}
