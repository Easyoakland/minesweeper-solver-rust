use ahash::AHasher;
#[allow(unused_imports)] // Lsb0 is indicated as unused but it is not unused.
use bitvec::{bitvec, order::Lsb0, vec::BitVec};
use cached::proc_macro::cached;
use captrs::{Bgr8, Capturer};
use enigo::{Enigo, MouseControllable};
use enum_iterator::{all, Sequence};
use image::{imageops, io, DynamicImage, GenericImageView, RgbImage};
use itertools::{Itertools, LendingIterator};
use std::{
    collections::HashMap,
    error::Error,
    fmt,
    fs::OpenOptions,
    hash::{BuildHasherDefault, Hash, Hasher},
    io::prelude::*,
    mem::take,
    ops::{Add, Sub},
};

pub type HashSet<T> = std::collections::HashSet<T, BuildHasherDefault<AHasher>>;

const TIMEOUTS_ATTEMPTS_NUM: u8 = 10;
const MAX_COMBINATIONS: u128 = 200_000;
const LOGGING: bool = false;

#[derive(Debug, Default, Copy, Clone, PartialEq)]
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

/// Error that indicates something went wrong while trying to solve the game.
/// These types of errors relate to the actual minesweeper game and not functional errors.
pub enum GameError {
    RevealedMine(CellCord),
    IncorrectFlag,
    UnidentifiedCell(CellCord),
    Unfinished,
    NoValidCombinations,
}

impl Error for GameError {}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::RevealedMine(x) => write!(f, "Revealed a mine at {x:?}."),
            Self::UnidentifiedCell(x) => write!(f, "Can't identify cell at {x:?}"),
            Self::Unfinished => write!(
                f,
                "The game was unable to be finished for an unknown reason."
            ),
            Self::IncorrectFlag => write!(f, "One of the flags is incorrect."),
            Self::NoValidCombinations => write!(f, "No valid combinations."),
        }
    }
}

/// Returns a dynamic image at the given path.
/// # Panics
/// Panics if the image can't be read.
/// Panics if the type of image is unsupported.
#[must_use]
pub fn read_image(path: &str) -> DynamicImage {
    io::Reader::open(path)
        .expect("Couldn't read image.")
        .decode()
        .expect("Unsupported Type")
}

pub fn save_image(path: &str, image: &DynamicImage) {
    save_rgb_vector(
        path,
        image.clone().into_rgb8().to_vec(),
        image.width(),
        image.height(),
    );
}

/// Returns a capturer instance. Selects monitor based upon passed id. Zero indexed.
/// Used primarily for benchmark.
/// # Errors
/// Returns the error that is given by underlying screen capture library.
pub fn setup_capturer(id: usize) -> Result<Capturer, String> {
    Capturer::new(id)
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
                } in frame
                {
                    rgb_vec.push(r);
                    rgb_vec.push(g);
                    rgb_vec.push(b);
                    /* rgb_vec.push(a); */
                }

                // Make sure the image is not a failed black screen.
                if !rgb_vec.iter().any(|&x| x != 0) {
                    continue;
                };
                return rgb_vec;
            }
            Err(_) => continue,
        }
    }
}

/// Captures and returns a screenshot as `RgbImage`.
pub fn capture_image_frame(capturer: &mut Capturer) -> RgbImage {
    RgbImage::from_raw(
        capturer.geometry().0,
        capturer.geometry().1,
        capture_rgb_frame(capturer),
    )
    .expect("Frame was unable to be captured and converted to RGB.")
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
/// # Panics
/// Panic if `super_width` or `super_height` can't be converted to i32 because returned Point uses i32 type.
#[must_use]
pub fn locate_all(super_image: &DynamicImage, sub_image: &DynamicImage) -> Vec<Point> {
    let super_image = super_image.clone().into_rgb8(); // `.clone().into_rgb8()` is 13% faster on benchmark then `.to_rgb8()`. As approx as fast.as_rgb.unwrap() (maybe 1% faster) but can use all of RgbImage methods this way.
    let sub_image = sub_image.clone().into_rgb8();
    let mut output = Vec::new();

    // Get dimensions of both images.
    let (super_width, super_height) = (super_image.width(), super_image.height());
    let (sub_width, sub_height) = (sub_image.width(), sub_image.height());

    // Iterate through all positions of the sup_image checking if that region matches the sub_image when cropped to size.
    let mut sub_super_image = imageops::crop_imm(&super_image, 0, 0, sub_width, sub_height);
    'y_loop: for y in 0..super_height {
        'x_loop: for x in 0..super_width {
            // Move sub image instead of creating a new one each iteration.
            // Approx 10% faster than using `let mut sub_sup_image = imageops::crop_imm(&sup_image, x, y, sub_width, sub_height);` each line
            sub_super_image.change_bounds(x, y, sub_width, sub_height);

            // Skip to next y line if no space left on this row for a sub image to fit.
            if x + sub_width > super_width {
                continue 'y_loop;
            }
            // Stop searching if no room for any more images.
            if y + sub_height > super_height {
                break 'y_loop;
            }

            // For each point if it doesn't match skip to next position.
            for (super_data, sub_data) in sub_super_image
                .pixels()
                .map(|(_x, _y, pixel)| pixel)
                .zip(sub_image.pixels())
            {
                if sub_data != &super_data {
                    continue 'x_loop;
                }
            }

            output.push(Point(x.try_into().unwrap(), y.try_into().unwrap()));
        }
    }
    output
}

fn exact_image_match(image1: &DynamicImage, image2: &DynamicImage) -> bool {
    // If the images don't have the same dimensions then they can't be an exact match.
    if (image1.width() != image2.width()) || (image1.height() != image2.height()) {
        return false;
    };

    // If they are the same dimension then locating all of either in the other will give one result if they are the same and 0 if different.
    let matches = locate_all(image1, image2);
    if matches.is_empty() {
        false
    } else if matches.len() == 1 {
        true
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
            if (i != j) && set1.is_subset(set2) {
                // Remove items that are shared from the superset.
                *set2 = set2.drain().filter(|elem| !set1.contains(elem)).collect();

                did_something = true;

                // Decrease the mine count of the superset by the subset's minecount.
                cell_group_vec[j].mine_num -= cell_group_vec[i].mine_num;
            }
        }
    }
    (did_something, cell_group_vec)
}

const CELL_VARIANT_COUNT: usize = 11;
#[derive(Debug, Copy, Clone, PartialEq, Eq, Sequence)]
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
    fn value(self) -> Option<u32> {
        match self {
            CellKind::One => Some(1),
            CellKind::Two => Some(2),
            CellKind::Three => Some(3),
            CellKind::Four => Some(4),
            CellKind::Five => Some(5),
            CellKind::Six => Some(6),
            CellKind::Seven => Some(7),
            CellKind::Eight => Some(8),
            CellKind::Flag | CellKind::Unexplored | CellKind::Explored => None,
        }
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
                "Invalid input. Can't convert {x} to a CellKind."
            ))),
        };
    }
}

/// Gets all factors of input.
/// ```
/// use minesweeper_solver_in_rust::prime_factorize;
/// assert_eq!(prime_factorize(5), vec![5]);
/// assert_eq!(prime_factorize(9), vec![3, 3]);
/// assert_eq!(prime_factorize(1201*1213), vec![1201, 1213]);
/// ```
#[must_use]
pub fn prime_factorize(mut n: u32) -> Vec<u32> {
    let mut out = vec![];

    for i in 2..(n + 1) {
        while n % i == 0 {
            n /= i;
            out.push(i);
        }
    }
    if out.is_empty() {
        out.extend_from_slice(&[1, n]);
    }

    out
}

/// Calculates from iter and saturates at bound.
#[must_use]
pub fn saturating_product<I: Iterator<Item = u32>>(iter: I) -> u64 {
    let mut out = 1u64;
    for item in iter {
        out = match out.checked_mul(u64::from(item)) {
            Some(x) => x,
            None => return u64::MAX,
        }
    }
    out
}

/// Simple combination calculation for positive values. Returns as float approximation.
/// If denominator saturates will get larger than expected value.
/// # Examples
/// ```
/// use minesweeper_solver_in_rust::saturating_combinations;
/// assert_eq!(saturating_combinations(5, 4), 5);
/// assert_eq!(saturating_combinations(12, 6), 924);
/// assert!(saturating_combinations(34, 17) > 2333600000 && saturating_combinations(34, 17) < 2333610000);
/// assert!(saturating_combinations(50, 25) == 126410606437752);
/// ```
#[must_use]
#[cached]
pub fn saturating_combinations(n: u32, r: u32) -> u128 {
    let mut numerator = (1..=n).flat_map(prime_factorize).collect::<Vec<u32>>();
    let mut denominator1 = (1..=r).flat_map(prime_factorize).collect::<Vec<u32>>();
    let mut denominator2 = (1..=(n - r))
        .flat_map(prime_factorize)
        .collect::<Vec<u32>>();

    // Remove replicated items in denominator1
    let mut i = 0;
    while i < numerator.len() {
        i += 1;
        if let Some(idx) = denominator1.iter().position(|&x| x == numerator[i - 1]) {
            if numerator.len() > 1 {
                numerator.swap_remove(i - 1);
            } else {
                numerator[0] = 1;
                break;
            }
            if denominator1.len() > 1 {
                denominator1.swap_remove(idx);
            } else {
                denominator1[0] = 1;
                break;
            }
            i -= 1;
        }
    }
    // Remove replicated items in denominator2
    let mut i = 0;
    while i < numerator.len() {
        i += 1;
        if let Some(idx) = denominator2.iter().position(|&x| x == numerator[i - 1]) {
            if numerator.len() > 1 {
                numerator.swap_remove(i - 1);
            } else {
                numerator[0] = 1;
                break;
            }
            if denominator2.len() > 1 {
                denominator2.swap_remove(idx);
            } else {
                denominator2[0] = 1;
                break;
            }
            i -= 1;
        }
    }

    u128::from(saturating_product(numerator.iter().copied()))
        / u128::from(
            saturating_product(denominator1.iter().copied())
                * saturating_product(denominator2.iter().copied()),
        )
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CellGroup {
    offsets: HashSet<usize>,
    mine_num: u32,
}

// Allow this clippy lint because the Hash derived by PartialEq does not work for `merge_overlapping_groups`.
// Also even if it did work the hash should not be based upon the mine_num for comparison of equality.
#[allow(clippy::derive_hash_xor_eq)]
impl Hash for CellGroup {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for offset in &self.offsets {
            offset.hash(state);
        }
    }
}

/// First item in inner vec is the item the rest of the items in the inner vec intersect to first order.
fn merge_overlapping_groups_first_order(input: &[CellGroup]) -> Vec<Vec<&CellGroup>> {
    let mut out = Vec::new();

    // Loop over input elements.
    for (k, e) in input.iter().enumerate() {
        let mut temp = Vec::new();

        temp.push(k);
        // Loop over second element compared to first.
        for (i, e2) in input.iter().enumerate() {
            // Don't re-include the same item.
            // If not disjoint include it in the local group.
            if i != k && !e.offsets.is_disjoint(&e2.offsets) {
                temp.push(i);
            }
        }

        // Each index that shared first_order_overlap
        let mut local_group = Vec::new();
        for j in temp {
            local_group.push(&input[j]);
        }
        out.push(local_group);
    }
    out
}

/// For each Subgroup if input element intersects then merge element into `Subgroup` and mark the `Subgroup`.
/// Then remove all marked Subgroup and combine before adding back.
/// If doesn't intersect make new Subgroup.
fn merge_overlapping_groups(input: &[CellGroup]) -> Vec<HashSet<&CellGroup>> {
    #[derive(Debug, Default)]
    struct Subgroup {
        offsets: HashSet<usize>,
        sets: HashSet<usize>,
    }

    let mut state: Vec<Subgroup> = Vec::new();
    for (i, e) in input.iter().enumerate() {
        let mut marks = vec![];
        let mut temp = vec![Subgroup {
            offsets: e.offsets.clone(),
            sets: {
                let mut set = HashSet::default();
                set.insert(i);
                set
            },
        }];

        // Merge input where not disjoint.
        for (k, t) in state.iter_mut().enumerate() {
            if !e.offsets.is_disjoint(&t.offsets) {
                marks.push(k);
                t.offsets.extend(e.offsets.clone());
            }
        }

        // Add all marked Subgroup to temp for future merge
        // Iterate in reverse so removal doesn't affect indexes.
        for &j in marks.iter().rev() {
            // Remove all marked and add into temp.
            temp.push(state.swap_remove(j));
        }

        // Merge items in temp into one Subgroup and add to state.
        let mut merged_group = Subgroup::default();
        for group in temp {
            merged_group.offsets.extend(group.offsets);
            merged_group.sets.extend(group.sets);
        }
        state.push(merged_group);
    }
    let mut out = vec![];
    for subgroup in state {
        let mut set = HashSet::default();
        for set_idx in subgroup.sets {
            set.insert(&input[set_idx]);
            // set.insert(set_idx); // inserts indexes in input instead of values
        }
        out.push(set);
    }
    out
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
            let x: i64 = cord.0 as i64 - radius as i64 + i as i64;
            let y: i64 = cord.1 as i64 - radius as i64 + j as i64;
            // Don't make neighbors with negative cords.
            if x < 0 || y < 0 {
                continue;
            }
            // If neither is negative can safely convert to unsigned.
            let x: usize = x.try_into().unwrap();
            let y: usize = y.try_into().unwrap();

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

    neighbors
}

fn cell_cord_to_offset(board_cell_width: u32, cord: CellCord) -> usize {
    let x_offset = cord.0;
    let y_offset = board_cell_width as usize * (cord.1);
    x_offset + y_offset
}

fn offset_to_cell_cord(board_cell_width: u32, mut offset: usize) -> CellCord {
    let mut cnt = 0;
    while offset >= board_cell_width as usize {
        offset -= board_cell_width as usize;
        cnt += 1;
    }
    let y = cnt;
    let x = offset;
    // y = int(Offset/self._width)
    // x = int(Offset-y*self._width)
    CellCord(x, y)
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
        neighbors_of_cord(self.cord, radius, board_cell_width, board_cell_height)
    }
}

#[derive(Debug)]
pub struct Simulation {
    state: BitVec,
    board_cell_width: u32,
    board_cell_height: u32,
}

impl Simulation {
    #[must_use]
    pub fn new(width: u32, height: u32, mine_num: u32, initial_guess: CellCord) -> Simulation {
        let mut state = BitVec::with_capacity(width as usize * height as usize);
        let neighbors = neighbors_of_cord(initial_guess, 1, width, height);
        let initial_guess_offset = cell_cord_to_offset(width, initial_guess);

        // Generate random unique indexes until there are enough to represent each mine.
        let mut mine_pos: Vec<u32> = vec![];
        while mine_pos.len() != mine_num as usize {
            let random_num = rand::random::<u32>() % (width * height);
            if !mine_pos.contains(&random_num)
                && initial_guess_offset != random_num as usize
                && !neighbors.contains(&offset_to_cell_cord(width, random_num as usize))
            {
                mine_pos.push(random_num);
            }
        }

        // Make a state where each position is not a mine unless its index matches one of the mine positions.
        for h in 0..height {
            for w in 0..width {
                state.push(mine_pos.contains(&(h * width + w)));
            }
        }
        Simulation {
            state,
            board_cell_width: width,
            board_cell_height: height,
        }
    }

    #[must_use]
    pub fn is_mine(&self, index: usize) -> bool {
        self.state[index]
    }

    /// Returns none if the item is a mine and otherwise returns how many mines are in the neighborhood.
    #[must_use]
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
                surrounding_mines += 1;
            }
        }
        Some(surrounding_mines)
    }
}

/// Holds the information related to the current state of the game.
#[derive(Default)]
pub struct Game {
    cell_images: Vec<RgbImage>,
    state: Vec<CellKind>,
    cell_positions: Vec<Point>,
    board_px_width: u32,
    board_px_height: u32,
    board_cell_width: u32,
    board_cell_height: u32,
    top_left: Point,
    individual_cell_width: u32,
    individual_cell_height: u32,
    board_screenshot: RgbImage,
    capturer: Option<Capturer>,
    frontier: Vec<Cell>,
    cell_groups: Vec<CellGroup>,
    simulation: Option<Simulation>,
    action_stack: Vec<CellCord>,
    mine_num: u32,
    endgame: bool,
}

impl Game {
    /// Creates a new game struct. Requires the input of the different cell type image file locations in order of CELL enum variants.
    /// Specifically the order is `One,Two,Three,Four,Five,Six,Seven,Eight,Flag,Unexplored,Explored`
    /// # Panics
    /// - If the given files are not readable and decodable it panics.
    /// - If a board is unable to be found on screen it panics.
    #[must_use]
    pub fn build(
        cell_files: [&str; CELL_VARIANT_COUNT],
        mine_num: u32,
        screenshot: RgbImage,
        capturer: Capturer,
    ) -> Game {
        // Reads in each cell image.
        let mut cell_images: Vec<RgbImage> = Vec::with_capacity(CELL_VARIANT_COUNT);
        for file in &cell_files {
            cell_images.push(
                io::Reader::open(file)
                    .unwrap_or_else(|e| {
                        panic!("Cell variant image \"{file}\" was unable to be read. {e}")
                    })
                    .decode()
                    .unwrap_or_else(|e| panic!("Unsupported Image type for \"{file}\". {e}"))
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
        assert!(
            !cell_positions.is_empty(),
            "Unable to find board. Unable to continue."
        );

        // Initializes state as all unexplored.
        let state = vec![CellKind::Unexplored; cell_positions.len()];

        let top_left = cell_positions[0];

        let individual_cell_width = cell_images[9].width();
        let individual_cell_height = cell_images[9].height();

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
        let px_width =
            <usize as TryInto<u32>>::try_into(cell_width).unwrap() * cell_images[9].width();
        // Same for height.
        let px_height =
            <usize as TryInto<u32>>::try_into(cell_height).unwrap() * cell_images[9].height();

        Game {
            cell_images,
            mine_num,
            state,
            cell_positions,
            board_px_width: px_width,
            board_px_height: px_height,
            board_cell_width: cell_width.try_into().unwrap(),
            board_cell_height: cell_height.try_into().unwrap(),
            top_left,
            individual_cell_width,
            individual_cell_height,
            capturer: Some(capturer),
            board_screenshot: screenshot, // Initially not cropped to just board. Will be when it is set using the correct method.
            frontier: Vec::new(),
            cell_groups: Vec::new(),
            simulation: None,
            action_stack: Vec::new(),
            ..Game::default()
        }
    }

    /// Used as friendlier interface to `Game::build` function. Handles setting up screen capture.
    /// # Panics
    /// If can't setup screen capture.
    #[must_use]
    pub fn new(cell_files: [&str; CELL_VARIANT_COUNT], mine_num: u32, id: usize) -> Game {
        let mut capturer = setup_capturer(id)
            .unwrap_or_else(|e| panic!("Can't setup a screen capturer because: {e}."));
        let screenshot = capture_image_frame(&mut capturer);
        Game::build(cell_files, mine_num, screenshot, capturer)
    }

    /// Sets up the `Game` so that it can run a simulation of solving a real game.
    ///
    /// Expert mode has the following values:
    /// ```
    /// let board_cell_width: u32 = 30;
    /// let board_cell_height: u32 = 16;
    /// let mine_num = 99;
    /// ```
    #[allow(clippy::missing_panics_doc)] // Uses unwrap that will never panic.
    #[must_use]
    pub fn new_for_simulation(
        board_cell_width: u32,
        board_cell_height: u32,
        mine_num: u32,
        initial_guess: CellCord,
    ) -> Game {
        // Initializes state as all unexplored.
        let state = vec![CellKind::Unexplored; (board_cell_height * board_cell_width) as usize];
        let simulation = Some(Simulation::new(
            board_cell_width,
            board_cell_height,
            mine_num,
            initial_guess,
        ));
        Game {
            simulation,
            mine_num,
            board_cell_height,
            board_cell_width,
            state,
            frontier: Vec::new(),
            cell_groups: Vec::new(),
            action_stack: Vec::new(),
            // Other parameters don't matter in simulation.
            ..Game::default()
        }
    }
    /// Sets the board screenshot of Game to just the board of tiles. Crops out extra stuff.
    fn get_board_screenshot_from_screen(&mut self) {
        let screenshot = capture_image_frame(
            self.capturer
                .as_mut()
                .expect("Tried getting a screenshot from the screen with no valid capturer."),
        );
        self.board_screenshot = image::imageops::crop_imm(
            &screenshot,
            self.top_left.0.try_into().unwrap(),
            self.top_left.1.try_into().unwrap(),
            self.board_px_width,
            self.board_px_height,
        )
        .to_image();
    }

    /// Left clicks indicated cord
    fn click_left_cell_cord(&self, cord: CellCord) {
        let mut enigo = Enigo::new();
        // Add extra 2 pixel so the click is definitely within the cell instead of maybe on the boundary.
        // Not entirely sure why this is necessary but it fixes a bug where the cell isn't clicked close enough to the center
        let x = self.cell_positions[cord.0].0 + 2;
        let y = self.cell_positions[cord.1 * self.board_cell_width as usize].1 + 2;
        enigo.mouse_move_to(x, y);
        enigo.mouse_down(enigo::MouseButton::Left);
        enigo.mouse_up(enigo::MouseButton::Left);
    }

    /// Right click indicated cord
    fn click_right_cell_cord(&self, cord: CellCord) {
        let mut enigo = Enigo::new();
        // Add extra 2 pixel so the click is definitely within the cell instead of maybe on the boundary.
        // Not entirely sure why this is necessary but it fixes a bug where the cell isn't clicked close enough to the center
        let x = self.cell_positions[cord.0].0 + 2;
        let y = self.cell_positions[cord.1 * self.board_cell_width as usize].1 + 2;
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
        self.cell_positions[offset]
    }

    fn state_at_cord(&mut self, cord: CellCord) -> &mut CellKind {
        let offset = self.cell_cord_to_offset(cord);
        &mut self.state[offset]
    }

    fn state_at_cord_imm(&self, cord: CellCord) -> &CellKind {
        let offset = self.cell_cord_to_offset(cord);
        &self.state[offset]
    }

    /// <p style="background:rgba(255,181,77,0.16);padding:0.75em;">
    /// <strong>Warning:</strong> Don't use this. It is only public to do a benchmark.
    /// </p>
    ///
    /// # Panics
    /// Panics if the `identify_cell` doesn't work on the CellCord(0, 0). Otherwise does nothing.
    pub fn identify_cell_benchmark_pub_func(&mut self) {
        self.identify_cell(CellCord(0, 0)).unwrap();
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
            TryInto::<u32>::try_into(pos.0).unwrap()
                - TryInto::<u32>::try_into(self.top_left.0).unwrap(),
            TryInto::<u32>::try_into(pos.1).unwrap()
                - TryInto::<u32>::try_into(self.top_left.1).unwrap(),
            self.individual_cell_width,
            self.individual_cell_height,
        );

        let sectioned_board_image = sectioned_board_image.to_image();
        // image must be dynamic image for exact_image_match function
        let sectioned_board_image = DynamicImage::from(sectioned_board_image);

        // Return the first cell type that matches the tile at the given cordinate.
        for (cell_image, cell_kind) in self.cell_images.iter().zip(all::<CellKind>()) {
            // See if it is a match.
            let is_match = exact_image_match(
                &DynamicImage::from(cell_image.clone()),
                &sectioned_board_image,
            );

            // If it is a match return that match and stop searching.
            if is_match {
                temp = cell_kind;
                return Ok(temp);
            }
        }

        // If all cell images were matched and none worked.
        if LOGGING {
            println!(
                "UNIDENTIFIED CELL at cord: {:?}(offset: {:?})",
                cord,
                self.cell_cord_to_offset(cord)
            );
        }
        save_image("cell_images/Unidentified.png", &sectioned_board_image);
        // If the program can't identify the cell then it shouldn't keep trying to play the game.
        self.save_state_info("test/FinalGameState.csv", false);
        Err(GameError::UnidentifiedCell(cord))
    }

    fn state_by_cell_cord(&mut self, cord: CellCord) -> &mut CellKind {
        let offset = self.cell_cord_to_offset(cord);
        &mut self.state[offset]
    }

    fn update_state(&mut self, cord: CellCord) -> Result<(), GameError> {
        // Only unexplored cells can update so don't bother checking if it wasn't an unexplored cell last time it was updated.
        let cell_state_record = *(self.state_by_cell_cord(cord));
        if cell_state_record != CellKind::Unexplored {
            return Ok(());
        }

        let cell = Cell {
            cord,
            kind: self.identify_cell(cord)?,
        };
        // If cell state is different from recorded for that cell.
        if cell.kind != cell_state_record {
            // Then update state record for that cell.
            *(self.state_by_cell_cord(cell.cord)) = cell.kind;
            // If the cell is fully explored then check it's neighbors because it is not part of the frontier and neighbors may have also updated.
            if cell.kind == CellKind::Explored {
                // Update state of its neighbors.
                for neighbor in cell.neighbors(1, self.board_cell_width, self.board_cell_height) {
                    self.update_state(neighbor)?;
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
                self.frontier.push(cell);
            }
        }
        Ok(())
    }

    pub fn reveal(&mut self, cord: CellCord) {
        self.click_left_cell_cord(cord);

        self.action_stack.push(cord);
        // self.update_state(cord);
    }

    fn process_action_stack(&mut self, simulate: bool) -> Result<(), GameError> {
        // Only need to check tha the most recent clicked cell is now updated. When it is presumably the previously clicked cells will have also updated.
        // Next block will repeatedly check if the most recent cell is updated.
        if !simulate {
            // If empty nothing to process so return.
            let Some(cord) = self.action_stack.last() else { return Ok(()) };
            let cord = *cord;

            let mut temp_cell_kind = CellKind::Unexplored;
            let mut i = 0;
            while temp_cell_kind == CellKind::Unexplored
            // This will keep looking for a change until one occurs.
            {
                // If it must wait this many times give up because something is wrong.
                assert!(i < TIMEOUTS_ATTEMPTS_NUM, "Board won't update. Game Loss?");

                self.get_board_screenshot_from_screen();
                temp_cell_kind = self.identify_cell(cord)?;
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
                        if self.identify_cell(neighbor)? == CellKind::Unexplored {
                            // If any neighbors are unexplored then this cell can't be a blank explored
                            temp_cell_kind = CellKind::Unexplored; // set a so while loop will take a new screenshot;
                            break 'neighbor_loop;
                        }
                    }
                    i += 1;
                }
            }
        }
        while !self.action_stack.is_empty() {
            let cord = self
                .action_stack
                .pop()
                .expect("While loop should prevent empty.");
            if !simulate {
                self.update_state(cord)?;
            } else if simulate {
                self.update_state_simulation(cord)?;
            };
        }
        Ok(())
    }

    fn identify_cell_simulation(&self, cord: CellCord) -> Result<CellKind, GameError> {
        let Some(value) = self
            .simulation
            .as_ref()
            .expect("Simulation doesn't exist but function requires it does.")
            .value(cell_cord_to_offset(self.board_cell_width, cord))
        else {
            return Err(GameError::RevealedMine(cord));
        };
        Ok(CellKind::value_to_cell_kind(value)
            .expect("Value should always be 0-8 from previous assignment."))
    }

    fn update_state_simulation(&mut self, cord: CellCord) -> Result<(), GameError> {
        // Only unexplored cells can update so don't bother checking if it wasn't an unexplored cell last time it was updated.
        let cell_state_record = *(self.state_by_cell_cord(cord));
        if cell_state_record != CellKind::Unexplored {
            return Ok(());
        }

        let cell = Cell {
            cord,
            kind: self.identify_cell_simulation(cord)?,
        };
        // If cell state is different from recorded for that cell.
        if cell.kind != cell_state_record {
            // Then update state record for that cell.
            *(self.state_by_cell_cord(cell.cord)) = cell.kind;
            // If the cell is fully explored then check it's neighbors because it is not part of the frontier and neighbors may have also updated.
            if cell.kind == CellKind::Explored {
                // Update state of its neighbors.
                for neighbor in cell.neighbors(1, self.board_cell_width, self.board_cell_height) {
                    self.update_state_simulation(neighbor)?;
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
                self.frontier.push(cell);
            }
        }
        Ok(())
    }

    fn reveal_simulation(&mut self, cord: CellCord) {
        self.action_stack.push(cord);
    }

    /// Flag cell at cord then update cell state at location to flag.
    fn flag(&mut self, cord: CellCord) {
        // Don't do anything if the cell isn't flaggable.
        if *self.state_at_cord_imm(cord) != CellKind::Unexplored {
            if LOGGING {
                println!("Tried flagging a non flaggable at: {cord:#?}");
            }
            return;
        }
        // Since the cell is flaggable, flag it...
        self.click_right_cell_cord(cord);
        // ...and update the internal state of that cell to match.
        *self.state_at_cord(cord) = CellKind::Flag;
        self.mine_num -= 1;
    }

    /// Flag cell at cord then update cell state at location to flag.
    fn flag_simulation(&mut self, cord: CellCord) {
        // Don't do anything if the cell isn't flaggable.
        if *self.state_at_cord_imm(cord) != CellKind::Unexplored {
            if LOGGING {
                println!("Tried flagging a non flaggable at: {cord:#?}");
            }
            return;
        }
        // Update the internal state of that cell to match.
        *self.state_at_cord(cord) = CellKind::Flag;
        self.mine_num -= 1;

        if LOGGING {
            let sim_mine_num = self
                .simulation
                .as_ref()
                .unwrap()
                .state
                .iter()
                .enumerate()
                .filter(|(i, x)| **x && self.state[*i] == CellKind::Unexplored)
                .count();
            if sim_mine_num != self.mine_num.try_into().unwrap() {
                {
                    self.save_state_info("test/FinalGameState.csv", true);
                    panic!("[Error] When flagging {cord:?} simulation has different mine num ({sim_mine_num:?}) then internal state ({})!", self.mine_num);
                }
            }
        }
    }

    /// Rule 1 implemented with sets. If the amount of mines in a set is the same as the amount of cells in a set, they are all mines.
    /// Returns a bool indicating whether the rule did something.
    fn cell_group_rule_1(&mut self, cell_group: &CellGroup, simulate: bool) -> bool {
        // If the number of mines in the set is the same as the size of the set.
        if cell_group.mine_num as usize == cell_group.offsets.len() {
            // Flag all cells in set.
            for offset in &cell_group.offsets {
                if !simulate {
                    self.flag(self.offset_to_cell_cord(*offset));
                } else if simulate {
                    self.flag_simulation(self.offset_to_cell_cord(*offset));
                }
            }
            // Rule activated.
            true
        } else {
            // Rule didn't activate.
            false
        }
    }

    fn cell_group_rule_2(&mut self, cell_group: &CellGroup, simulate: bool) -> bool {
        // If set of cells has no mine. (mine_num is 0 from previous if)
        if cell_group.mine_num == 0 {
            // The reveals at the end might affect cells that have yet to be changed.
            for offset in &cell_group.offsets {
                // If a previous iteration of this loop didn't already reveal that cell, then reveal that cell.
                if self.state[*offset] == CellKind::Unexplored {
                    if !simulate {
                        self.reveal(self.offset_to_cell_cord(*offset));
                    } else if simulate {
                        self.reveal_simulation(self.offset_to_cell_cord(*offset));
                    }
                }
            }
            true // Rule activated.
        } else {
            false // Didn't activate.
        }
    }

    // Generates a set for a given cell.
    fn generate_cell_group(&self, cell: &Cell) -> Option<CellGroup> {
        // Make a set for all locations given as an offset.
        let mut offsets = HashSet::default();
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
                flag_cnt += 1;
            }
        }
        // If set is empty don't return anything because there is no valid CellGroup
        if offsets.is_empty() {
            return None;
        }

        // Set mine num based on the cell's number.
        if let Some(cell_value) = cell.kind.value() {
            // The amount of mines in the CellGroup is the amount there are around the cell minus how many have already been identified
            let mine_num = cell_value.saturating_sub(flag_cnt);
            Some(CellGroup { offsets, mine_num })
        }
        // If the cell doesn't have a number then return nothing because non-numbered cells don't have associated CellGroup.
        else {
            None
        }
    }

    fn process_frontier(&mut self, simulate: bool) {
        while !self.frontier.is_empty() {
            let current_cell = self
                .frontier
                .pop()
                .expect("Already checked frontier length > 0.");
            let cell_group = self.generate_cell_group(&current_cell);

            if let Some(cell_group) = cell_group {
                // If rule 1 was able to do something currentCell.
                // Then the rest of the loop is unnecessary.
                // The short circuit comparison means the first rule to do something will terminate the comparison.
                // Both rules should not be able to activate.
                if self.cell_group_rule_1(&cell_group, simulate)
                    || self.cell_group_rule_2(&cell_group, simulate)
                {
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
    fn deterministic_solve(&mut self, simulate: bool) -> Result<(), GameError> {
        // Makes outermost loop always execute at least once.
        let mut do_while_flag = true;
        // Loops through frontier and self.cell_groups.
        // Continues looping until inner loop indicates self.cell_groups can't be processed anymore and outer loop indicates the frontier is empty.
        while do_while_flag || !self.frontier.is_empty() {
            do_while_flag = false;
            while !self.frontier.is_empty() || !self.action_stack.is_empty() {
                self.process_action_stack(simulate)?;
                self.process_frontier(simulate);
            }
            // Set did_someting to 1 so self.cell_groups is processed at least once.
            let mut did_something = 1;
            while did_something > 0 && !self.cell_groups.is_empty() {
                // Set flag to 0 so it will detect no changes as still being 0.
                did_something = 0;

                // Simplify any CellGroup that can be simplified.
                // Following is a for loop of self.cell_groups.len() but the index is changed in some places in the loop so spots aren't missed.
                let mut i = 0;
                // for i in 0..self.cell_groups.len() {
                while i < self.cell_groups.len() {
                    // TODO split to function START -----------------------------------------------------------------------------------------
                    // Check to see if any cell_group now contain a flag or an already explored cell.
                    for offset in self.cell_groups[i].offsets.clone() {
                        // If the cell_group now contain a flag.
                        if self.state[offset] == CellKind::Flag {
                            // Remove the flag from the cell_group...
                            assert!(
                                self.cell_groups[i].offsets.remove(&offset),
                                "Removed offset: {:?} that wasn't in cell_group: {:?}.",
                                offset,
                                self.cell_groups[i]
                            );
                            // ...and decrease the amount of mines left.
                            if let Some(new_mine_num) = self.cell_groups[i].mine_num.checked_sub(1)
                            {
                                self.cell_groups[i].mine_num = new_mine_num;
                            } else {
                                // If there is an error when subtracting it means that a flag was incorrect at some point and now the board can have inconsistency.
                                // I imagine there is a way to recover from this but it is rather complicated if possible.
                                return Err(GameError::IncorrectFlag);
                            }
                            did_something += 1;
                        }
                        // If the cell_group now contains an not unexplored cell remove that cell as it can't be one of the mines anymore.
                        else if self.state[offset] != CellKind::Unexplored {
                            self.cell_groups[i].offsets.remove(&offset);
                            did_something += 1;
                        }

                        // Below will occur if there is an inconsistent board state from an incorrect flag.
                        if self.cell_groups[i].mine_num as usize > self.cell_groups[i].offsets.len()
                        {
                            return Err(GameError::IncorrectFlag);
                            // DEBUG
                            // self.save_state_info("test/FinalGameState.csv", simulate);
                            // panic!("ERROR at self.cell_groups[{i}]={:?} has more mines than cells to fill. Just removed {offset}",self.cell_groups[i]);
                        }
                    }
                    // TODO split to function END -----------------------------------------------------------------------------------------

                    // TODO split to function START -----------------------------------------------------------------------------------------
                    // Check if a logical operation can be done.
                    let cell_groups = self.cell_groups[i].clone();

                    // Short circuit because only one rule should be able to work at a time and saves computation.
                    // Also makes sure loop does one rule per iteration.
                    if self.cell_group_rule_1(&cell_groups, simulate)
                        || self.cell_group_rule_2(&cell_groups, simulate)
                    {
                        // Since that cell_group is solved it is no longer needed.
                        self.cell_groups.swap_remove(i);
                        // Decrement loop index so this index is not skipped in next iteration now that a new value is in the index's position.
                        i = usize::saturating_sub(i, 1); // Saturate so a 0 index will terminate on next loop.
                        did_something += 1;
                    }
                    // Increment loop index
                    i += 1;
                    // TODO split to function END -----------------------------------------------------------------------------------------
                }

                // Remove subset-superset overlaps.
                {
                    let overlaps_existed;
                    (overlaps_existed, self.cell_groups) =
                        remove_complete_cell_group_overlaps(take(&mut self.cell_groups));
                    if overlaps_existed {
                        did_something += 1;
                    }
                }
            }
        }
        Ok(())
    }

    #[must_use]
    fn enumerate_all_possible_arrangements<'a>(
        sub_group_mine_num_lower_limit: usize,
        sub_group_mine_num_upper_limit: usize,
        sub_group_total_offsets_after_overlaps_removed: &HashSet<usize>,
        sub_group: impl IntoIterator<Item = &'a &'a CellGroup>,
        validator: impl Fn(u32, &CellGroup) -> bool,
    ) -> (i32, HashMap<usize, i32>) {
        // Iterate through all possible amounts of mines in the subgroup.
        // Calculate odds as the amount of times a mine appeared in a position divided by number of valid positions.
        // Division is done because comparison at the end will include other probabilities with different denominators.
        // Holds how often a mine occurs in each offset position.
        let mut occurrences_of_mine_per_offset: HashMap<_, _> =
            sub_group_total_offsets_after_overlaps_removed
                .iter()
                .map(|offset| (*offset, 0))
                .collect();
        let offset_to_bit_pos: HashMap<_, _> = sub_group_total_offsets_after_overlaps_removed
            .iter()
            .enumerate()
            .map(|(i, offset)| (*offset, i))
            .collect();
        let bit_pos_to_offset: HashMap<_, _> = sub_group_total_offsets_after_overlaps_removed
            .iter()
            .enumerate()
            .map(|(i, offset)| (i, *offset))
            .collect();

        // Must fit in a usize to do this method. For expert mode these values shouldn't be reached.
        // TODO fallback to previous method for larger.
        assert!(offset_to_bit_pos.len() <= usize::BITS.try_into().unwrap());
        assert!(sub_group_mine_num_upper_limit <= usize::BITS.try_into().unwrap());

        let cell_group_bitmasks = {
            let mut out = HashSet::default();
            for cell_group in sub_group {
                let mut temp = 0usize;
                for offset in &cell_group.offsets {
                    temp |= 1 << offset_to_bit_pos[offset];
                }
                out.insert((*cell_group, temp));
            }
            out
        };
        let mut number_of_valid_combinations = 0;
        for sub_group_mine_num in
            1.max(sub_group_mine_num_lower_limit)..(sub_group_mine_num_upper_limit + 1)
        {
            // Count up how many times a mine occurs in each offset position.
            // TODO Generating and allocating vectors to store each combination takes up 27%-36% of runtime. A faster implementation should be used.
            let mut combinations_iter = (0..(sub_group_total_offsets_after_overlaps_removed.len()))
                .combinations_lending(sub_group_mine_num);
            'combinations: while let Some(combination) = combinations_iter.next() {
                // Initialize the mask and set the positions that are selected this combination as ones.
                let mut combination_bitmask = 0usize;
                (combination.clone()).for_each(|bit_pos| {
                    combination_bitmask |= 1 << bit_pos;
                });

                // Verifies that the number of mines in this combination is not invalid for each individual CellGroup
                for (cell_group, bitmask) in &cell_group_bitmasks {
                    // Stores how many mines are in a CellGroup for this particular arrangement of mines.
                    let individual_cell_group_mine_num_for_specific_combination = {
                        // Should really do an and operation here but & is not defined for FixedBitSet. cloning and assigning with that is slower by ~30% (32 vs 47)
                        (combination_bitmask & bitmask).count_ones()
                    };
                    // If the amount of mines isn't the right amount.
                    // Go to the next combination because this one doesn't work.
                    if !validator(
                        individual_cell_group_mine_num_for_specific_combination,
                        cell_group,
                    ) {
                        continue 'combinations;
                    }
                }
                // Since the amount of mines is correct.
                // For every offset in this valid combination.
                // Increment the amount of mines at each offset.
                (combination).for_each(|bit_pos| {
                    *(occurrences_of_mine_per_offset
                        .get_mut(&bit_pos_to_offset[&bit_pos])
                        .unwrap()) += 1;
                });

                // Since the arrangement is valid increment the number of valid arrangements.
                number_of_valid_combinations += 1;
            }
        }
        (number_of_valid_combinations, occurrences_of_mine_per_offset)
    }

    fn update_likelihoods_from_enumerated_arrangements(
        &self,
        most_likely_positions: &mut Vec<usize>,
        most_likelihood: &mut f64,
        least_likely_positions: &mut Vec<usize>,
        least_likelihood: &mut f64,
        number_of_valid_combinations: i32,
        occurrences_of_mine_per_offset: &HashMap<usize, i32>,
    ) {
        // Enumerate offsets and chances of those offsets.
        for (offset, occurrence_of_mine_at_offset) in occurrences_of_mine_per_offset.iter() {
            // The chance a mine is somewhere is the amount of combinations a mine occurred in that position divided by how many valid combinations there are total.
            let chance_of_mine_at_position: f64 =
                f64::from(*occurrence_of_mine_at_offset) / f64::from(number_of_valid_combinations);

            if chance_of_mine_at_position > *most_likelihood {
                // If likelyhood of mine is higher than previously recorded.
                // Update likelyhood
                // and update position with highest likelyhood.
                *most_likelihood = chance_of_mine_at_position;
                *most_likely_positions = vec![*offset];
            }
            // If the likelyhood is 100% then add it anyway because 100% means theres a mine for sure and it should be flagged regardless.
            // It is better to miss 100% to floating point error than to generate an incorrect 100%.
            else if f64::abs(chance_of_mine_at_position - 1.0) < f64::EPSILON {
                // update likelyhood and append position of garrunteed mine.
                *most_likelihood = 1.0;
                most_likely_positions.push(*offset);
            }
            // If neither larger than previous nor garrunteed mine then check if new is is a corner position. If yes then that is new guess target. Supposedly this increases winrate.
            else if f64::abs(chance_of_mine_at_position - *most_likelihood)
                <= chance_of_mine_at_position.abs().max(most_likelihood.abs()) * f64::EPSILON
            {
                let cord = self.offset_to_cell_cord(*offset);
                // New item is corner.
                if (cord.0 == 0 || cord.0 == (self.board_cell_width - 1).try_into().unwrap())
                    && (cord.1 == 0 || cord.1 == (self.board_cell_height - 1).try_into().unwrap())
                {
                    *most_likelihood = chance_of_mine_at_position;
                    *most_likely_positions = vec![*offset];
                }
            }
            // Same thing but for leastlikelyhood.
            if chance_of_mine_at_position < *least_likelihood {
                *least_likelihood = chance_of_mine_at_position;
                *least_likely_positions = vec![*offset];
            }
            // If the chance of a mine is zero then it is guaranteed to not have a mine and should be revealed regardless.
            // Better to miss actual 0% from floating point error than to generate incorrect 0%
            else if chance_of_mine_at_position.abs()
                < chance_of_mine_at_position.abs() * f64::EPSILON
            {
                *least_likelihood = 0.0;
                least_likely_positions.push(*offset);
            }
            // If neither larger than previous nor garrunteed mine then check if new is is a corner position. If yes then that is new guess target. Supposedly this increases winrate.
            else if f64::abs(chance_of_mine_at_position - *least_likelihood)
                <= chance_of_mine_at_position.abs().max(least_likelihood.abs()) * f64::EPSILON
            {
                let cord = self.offset_to_cell_cord(*offset);
                // New item is corner.
                if (cord.0 == 0 || cord.0 == (self.board_cell_width - 1).try_into().unwrap())
                    && (cord.1 == 0 || cord.1 == (self.board_cell_height - 1).try_into().unwrap())
                {
                    *least_likelihood = chance_of_mine_at_position;
                    *least_likely_positions = vec![*offset];
                }
            }
        }
    }

    fn act_on_probabilities(
        &mut self,
        most_likelihood: f64,
        most_likely_positions: Vec<usize>,
        least_likelihood: f64,
        least_likely_positions: Vec<usize>,
        simulate: bool,
        did_something: &mut u32,
    ) {
        // If know where a mine or multiple are with certainty then flag each place that is definitely a mine.
        if f64::abs(most_likelihood - 1.0) <= f64::EPSILON {
            for most_likely_position in most_likely_positions {
                if LOGGING {
                    println!(
                        "Flagging {:?} with odds {:?} of being mine.",
                        self.offset_to_cell_cord(most_likely_position),
                        most_likelihood
                    );
                    if simulate {
                        if self
                            .simulation
                            .as_ref()
                            .unwrap()
                            .is_mine(most_likely_position)
                        {
                            println!("This is correct for the simulation");
                        } else {
                            println! {"This is incorrect for the simulation"};
                        }
                    }
                }
                let cord = self.offset_to_cell_cord(most_likely_position);
                if !simulate {
                    self.flag(cord);
                } else if simulate {
                    self.flag_simulation(cord);
                }
                *did_something += 1;
            }
        }
        // If there wasn't a spot that had a definite mine.
        // Then reveal all spots with lowest odds of mine.
        else {
            for least_likely_position in least_likely_positions {
                if LOGGING {
                    println!(
                        "Revealing {:?} with odds {:?} of being mine",
                        self.offset_to_cell_cord(least_likely_position),
                        least_likelihood
                    );
                }
                if !simulate {
                    self.reveal(self.offset_to_cell_cord(least_likely_position));
                } else if simulate {
                    self.reveal_simulation(self.offset_to_cell_cord(least_likely_position));
                }
            }
            *did_something += 1;
        }
    }

    /// First order intersection solve. Identifies 100% and 0% mine locations by iterating all possibilities of single cells compared to it's neighbors.
    fn local_solve(&mut self, simulate: bool) -> u32 {
        let mut did_something = 0;

        // Generate first order intersections for each `CellGroup`
        let local_groups = merge_overlapping_groups_first_order(&self.cell_groups);

        let mut reveal_pos = vec![];
        let mut flag_pos = vec![];

        // For each of these groups
        for group in local_groups {
            let group_total_offsets_no_overlap = {
                let mut temp = HashSet::default();
                for &x in &group {
                    x.offsets.iter().for_each(|&y| {
                        temp.insert(y);
                    });
                }
                temp
            };

            // Enumerate all possibilities for the group finding how often mines appear in each location.
            // Could be `enumerate_all_arrangements` but this is faster in benchmarks.
            let (number_of_valid_combinations, occurrences_of_mine_per_offset) = {
                let mut occurrences_of_mine_per_offset: HashMap<_, _> =
                    group_total_offsets_no_overlap
                        .iter()
                        .map(|&offset| (offset, 0))
                        .collect();
                let mut number_of_valid_combinations = 0;

                // For each possible arrangement of mines for the particular cell that is being examined.
                'combinations: for combination in group[0]
                    .offsets
                    .iter()
                    .combinations(group[0].mine_num.try_into().unwrap())
                {
                    // Count up how many times a mine occurs in each offset position.
                    for cell_group in &group {
                        // Stores how many mines are in `cell_group` for this particular arrangement of mines.
                        let mut individual_cell_group_mine_num_for_specific_combination = 0;
                        // For every offset in `cell_group`.
                        for offset in &cell_group.offsets {
                            // If the offset is a mine.
                            if combination.contains(&offset) {
                                // Increment how many mines are in `cell_group` for this arrangement.
                                individual_cell_group_mine_num_for_specific_combination += 1;
                            }
                        }
                        // Verifies that the number of mines left to find for the `CellGroup`
                        // is not more than the amount of spots it has left to fill.
                        if cell_group.mine_num.saturating_sub(individual_cell_group_mine_num_for_specific_combination)
                            > cell_group.offsets.difference(&group[0].offsets).count().try_into().unwrap()
                            // Verifies that the number of mines placed in the `CellGroup` is not greater the `CellGroup` mine_num
                            || individual_cell_group_mine_num_for_specific_combination
                            > cell_group.mine_num
                        {
                            continue 'combinations;
                        }
                    }

                    // Since the amount of mines is correct.
                    // For every offset in this valid combination
                    // increment the amount of mines at each offset.
                    for offset in combination {
                        *(occurrences_of_mine_per_offset.get_mut(offset).unwrap()) += 1;
                    }

                    // Since the arrangement is valid increment the number of valid arrangements.
                    number_of_valid_combinations += 1;
                }

                (number_of_valid_combinations, occurrences_of_mine_per_offset)
            };

            // Check each position of the particular cell being examined.
            for &offset in &group[0].offsets {
                let occurrence_cnt = occurrences_of_mine_per_offset[&offset];
                // If no mine occurred in a particular position flag it.
                if occurrence_cnt == 0 {
                    reveal_pos.push(offset);
                    did_something += 1;
                }
                // Else if mine always occurred in particular position reveal it.
                else if occurrence_cnt == number_of_valid_combinations {
                    flag_pos.push(offset);
                    did_something += 1;
                }
            }
        }

        // Reveal and execute here because immutable on self would have prevented it in loop.
        {
            for offset in reveal_pos {
                let cord = self.offset_to_cell_cord(offset);
                if !simulate {
                    self.reveal(cord);
                } else if simulate {
                    self.reveal_simulation(cord);
                }
            }
            for offset in flag_pos {
                let cord = self.offset_to_cell_cord(offset);
                if !simulate {
                    self.flag(cord);
                } else if simulate {
                    self.flag_simulation(cord);
                }
            }
        }

        did_something
    }

    fn calc_group_mine_limits(
        &self,
        sub_group: &HashSet<&CellGroup>,
        sub_group_num: usize,
    ) -> (usize, usize, HashSet<usize>) {
        let mut sub_group_total_offsets: Vec<usize> = Vec::new();
        let mut sub_group_mine_num_upper_limit_for_completely_unshared_mines = 0;

        // Put all offsets of corresponding subgroup into a Vec. Also count how many mines exist if there are no duplicates.
        for cell_group in sub_group {
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

        // An upper limit of mines is the number of positions in the sub_group. It can't have more mines than it has positions.
        // Another is if every intersection does not have a mine.
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

        // Actual upper limit can also be limited by the fact that the amount of mines in a subgroup must be at least 1.
        // The +1 below is because the number of sub groups - 1 is the number of other groups that have a bomb not in this sub_group.
        let sub_group_mine_num_upper_limit = sub_group_mine_num_upper_limit
            .min(<u32 as TryInto<usize>>::try_into(self.mine_num).unwrap() - sub_group_num + 1);

        (
            sub_group_mine_num_lower_limit,
            sub_group_mine_num_upper_limit,
            sub_group_total_offsets_after_overlaps_removed,
        )
    }

    /// Make best guess from all possibilities.
    /// # Panics
    /// If it can't find a valid move then a divide by zero will occur. This should panic.
    fn probabilistic_guess(&mut self, simulate: bool) -> u32 {
        let mut did_something = 0;

        // Keep track of the most and least likely places for there to be a mine and the likelyhood of each.
        let mut most_likely_positions = Vec::new();
        let mut most_likelihood = -1.0;
        let mut least_likely_positions = Vec::new();
        let mut least_likelihood = 101.0;

        // Find the sub groups of the grid of interconnected cell_groups that are not related or interconnected.
        // Basically partitions board so parts that don't affect each other are handled separately to make the magnitudes of the combinations later on more managable.
        let sub_groups = merge_overlapping_groups(&self.cell_groups);
        let sub_group_num = sub_groups.len();

        // For each independent sub group of cell_groups.
        for sub_group in sub_groups {
            fn validator(
                individual_cell_group_mine_num_for_specific_combination: u32,
                cell_group: &CellGroup,
            ) -> bool {
                // If the amount of mines isn't the right amount.
                // Go to the next combination because this one doesn't work.
                individual_cell_group_mine_num_for_specific_combination == cell_group.mine_num
            }

            let (
                sub_group_mine_num_lower_limit,
                sub_group_mine_num_upper_limit,
                sub_group_total_offsets_after_overlaps_removed,
            ) = self.calc_group_mine_limits(&sub_group, sub_group_num);

            // Check that the amount of combinations will not exceed the global variable for the maximum combinations.
            // If it does this will take too long.
            let mut combination_total = 0;
            // From the least to the most possible number of mines.
            for sub_group_mine_num in
                1.max(sub_group_mine_num_lower_limit)..(sub_group_mine_num_upper_limit + 1)
            {
                // Calculate the amount of combinations.
                let combination_amount = saturating_combinations(
                    sub_group_total_offsets_after_overlaps_removed
                        .len()
                        .try_into()
                        .unwrap(),
                    sub_group_mine_num.try_into().unwrap(),
                );
                combination_total += combination_amount;

                // If there are too many combinations then abort calculation.
                if combination_total > MAX_COMBINATIONS {
                    if LOGGING {
                        println!("Not computing {combination_total} total combinations.");
                        unimplemented!() // TODO remove this after handling this possibility with a fast guess method.
                    }
                    // Return that nothing was done.
                    return 0;
                }
            }

            let (number_of_valid_combinations, occurrences_of_mine_per_offset) =
                Game::enumerate_all_possible_arrangements(
                    sub_group_mine_num_lower_limit,
                    sub_group_mine_num_upper_limit,
                    &sub_group_total_offsets_after_overlaps_removed,
                    &sub_group,
                    validator,
                );
            if number_of_valid_combinations <= 0 {
                self.save_state_info("test/FinalGameState.csv", simulate);
                dbg!(sub_group_mine_num_lower_limit);
                dbg!(sub_group_mine_num_upper_limit);
                dbg!(sub_group);
                panic!("There were no valid combinations!")
            }
            self.update_likelihoods_from_enumerated_arrangements(
                &mut most_likely_positions,
                &mut most_likelihood,
                &mut least_likely_positions,
                &mut least_likelihood,
                number_of_valid_combinations,
                &occurrences_of_mine_per_offset,
            );
        }

        self.act_on_probabilities(
            most_likelihood,
            most_likely_positions,
            least_likelihood,
            least_likely_positions,
            simulate,
            &mut did_something,
        );

        did_something
    }

    /// Solves the game. Take a boolean indicating whether it is a simulated game or not.
    /// # Panics
    /// Primarily panics if the given starting cordinate doesn't exist in the game.
    /// Also panics if there is an internal bug in the code which is detected in one of the sub functions.
    /// # Errors
    /// Returns a `Err(GameError)` corresponding to an issue solving the game if one occurs.
    pub fn solve(&mut self, initial_guess: CellCord, simulate: bool) -> Result<(), GameError> {
        assert!(
            initial_guess.0 <= self.board_cell_width as usize,
            "Initial guess is larger than the board width."
        );
        assert!(
            initial_guess.1 <= self.board_cell_height as usize,
            "Initial guess is larger than the board height."
        );

        if !simulate {
            // Reveal initial tile.
            self.reveal(initial_guess);
        } else if simulate {
            self.reveal_simulation(initial_guess);
        }

        let mut did_something = 1;
        while did_something > 0 || !self.action_stack.is_empty() {
            // Did something is set to 0 so the loop will only continue is something happens to change it.
            did_something = 0;

            // Should always try deterministic solve at least once in case the previous guess made deterministic solving possible through a flag.
            // If it did then there would be no new actions on the action_stack but there might be something new that can be done.
            // At the very least deterministic solve will remove the now flagged location from all the `self.cell_groups`
            self.deterministic_solve(simulate)?;

            while !self.action_stack.is_empty() {
                // Loop will also continue if the state of the board is about to be updated and therefore there might be new moves.
                self.process_action_stack(simulate)?;
                self.deterministic_solve(simulate)?;
            }

            // Attempt local solves
            if self.local_solve(simulate) > 0 {
                // If local solves occurred then don't go to guess.
                continue;
            };

            if !self.cell_groups.is_empty() {
                if LOGGING {
                    print!("Guess required. ");
                }
                self.process_action_stack(simulate)?;

                // Handle endgame case. If unexplored is small enough add them as a cell_group for more potential deterministic solutions.
                if !self.endgame {
                    let unexplored_count = self
                        .state
                        .iter()
                        .filter(|&&x| x == CellKind::Unexplored)
                        .count();
                    // If the number of combinations added by the endgame cell group is less than the max combinations then add it.
                    if saturating_combinations(unexplored_count.try_into().unwrap(), self.mine_num)
                        <= MAX_COMBINATIONS
                    {
                        self.cell_groups.push(CellGroup {
                            offsets: self
                                .state
                                .iter()
                                .enumerate()
                                .filter(|(_, &x)| x == CellKind::Unexplored)
                                .map(|(offset, _)| offset)
                                .collect::<HashSet<_>>(),
                            mine_num: self.mine_num,
                        });
                        self.endgame = true;
                        if LOGGING {
                            println!(
                                "Added endgame cell_group: {:?}",
                                self.cell_groups.last().unwrap()
                            );
                        }
                        // Might be able to do deterministic solution now that endgame cell group was added.
                        did_something += 1;
                        continue;
                    }
                }
                if self.probabilistic_guess(simulate) >= 1 {
                    did_something += 1;
                    continue;
                }
            }
        }

        // If the game still contains unexplored cells it was not fully solved.
        if self.state.contains(&CellKind::Unexplored) {
            // This situation can occur if a wall of mines separates the two section of the board because the frontier won't be able to pass the wall.
            // So start solution at a random unexplored cell which most likely be on the other side of the mine wall.
            self.solve(
                offset_to_cell_cord(
                    self.board_cell_width,
                    self.state
                        .iter()
                        .position(|&x| x == CellKind::Unexplored)
                        .unwrap(),
                ),
                simulate,
            )
        } else {
            Ok(())
        }
    }

    /// Saves information about this Game to file for potential debugging purposes.
    /// # Panics
    /// If it can't save to file path specified.
    pub fn save_state_info(&self, path: &str, simulate: bool) {
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
                eprintln!("Couldn't write to file: {e}");
            }
            if (i + 1) % self.board_cell_width as usize == 0 {
                if let Err(e) = writeln!(file) {
                    eprintln!("Couldn't write to file: {e}");
                }
            }
        }

        if simulate {
            // Separate simulation from solution attempt with space.
            if let Err(e) = writeln!(file) {
                eprintln!("Couldn't write to file: {e}");
            }

            for (i, cell) in (0..self.simulation.as_ref().unwrap().state.len())
                .map(|i| self.simulation.as_ref().unwrap().value(i))
                .enumerate()
            {
                let symbol_to_write = match cell {
                    None => 'F',
                    Some(c) => match c {
                        0 => 'E',
                        1..=8 => char::from_digit(c, 10).unwrap(),
                        _ => panic!("c was {c} which is not a valid amount of mines."),
                    },
                };
                if let Err(e) = write!(file, "{symbol_to_write} ") {
                    eprintln!("Couldn't write to file: {e}");
                }
                if (i + 1) % self.board_cell_width as usize == 0 {
                    if let Err(e) = writeln!(file) {
                        eprintln!("Couldn't write to file: {e}");
                    }
                }
            }
        }

        // Separate with space
        if let Err(e) = writeln!(file) {
            eprintln!("Couldn't write to file: {e}");
        }
        if let Err(e) = write!(file, "{:#?}", self.cell_groups) {
            eprintln!("Couldn't write to file: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ignore = "Requires manual confirmation."]
    #[test]
    fn record_screen_to_file() {
        let mut capturer = setup_capturer(0).expect("Could not get a valid capturer.");
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

    #[ignore = "Requires manual confirmation and having open game on screen first."]
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
            99,
            0,
        );

        game.click_left_cell_cord(CellCord(1, 1));
    }

    #[ignore = "Requires manual confirmation and having open game on screen first."]
    #[test]
    fn reveal() {
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
            99,
            0,
        );
        let place_to_click = CellCord(7, 7);
        game.get_board_screenshot_from_screen();
        game.reveal(place_to_click);

        game.save_state_info("test/FinalGameState.csv", false);
        // First move always gives a fully unexplored cell.
        assert_eq!(*game.state_at_cord(place_to_click), CellKind::Explored);
    }

    #[ignore = "Requires manual confirmation."]
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
            99,
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
            99,
            read_image("test_in/subimage_search/board.png").to_rgb8(),
            setup_capturer(0).expect("Could not get a valid capturer."),
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
                offsets: HashSet::from_iter([1, 2, 3, 4, 5].into_iter()),
            },
            CellGroup {
                mine_num: 2,
                offsets: HashSet::from_iter([1, 2, 3, 4].into_iter()),
            },
        ];
        let (result_bool, result_of_no_overlap) =
            remove_complete_cell_group_overlaps(cell_group_vec);
        assert!(result_bool);
        assert!(result_of_no_overlap.contains(&CellGroup {
            offsets: HashSet::from_iter([4, 2, 1, 3].into_iter()),
            mine_num: 2,
        }));
        assert!(result_of_no_overlap.contains(&CellGroup {
            offsets: HashSet::from_iter([5].into_iter()),
            mine_num: 1,
        }));
    }

    #[test]
    fn merge_overlapping_groups_test() {
        let cell1 = CellGroup {
            offsets: HashSet::from_iter([1, 2].into_iter()),
            mine_num: 1,
        };
        let cell2 = CellGroup {
            offsets: HashSet::from_iter([1, 2, 3].into_iter()),
            mine_num: 2,
        };
        let cell3 = CellGroup {
            offsets: HashSet::from_iter([8, 9, 10].into_iter()),
            mine_num: 2,
        };
        let groups = vec![cell1.clone(), cell2.clone(), cell3.clone()];
        let output = merge_overlapping_groups(&groups);
        assert!(&output[0].contains(&cell1));
        assert!(&output[0].contains(&cell2));
        assert!(&output[1].contains(&cell3));

        let groups = vec![
            CellGroup {
                offsets: HashSet::from_iter([248, 249].into_iter()),
                mine_num: 1,
            },
            CellGroup {
                offsets: HashSet::from_iter([278, 248].into_iter()),
                mine_num: 1,
            },
            CellGroup {
                offsets: HashSet::from_iter([279, 249].into_iter()),
                mine_num: 1,
            },
            CellGroup {
                offsets: HashSet::from_iter([279, 278].into_iter()),
                mine_num: 1,
            },
        ];
        let output = merge_overlapping_groups(&groups);
        assert_eq!(output, vec![groups.iter().collect::<HashSet<_>>()]);

        let groups = vec![
            CellGroup {
                offsets: HashSet::from_iter([1, 2].into_iter()),
                mine_num: 1,
            },
            CellGroup {
                offsets: HashSet::from_iter([5, 4].into_iter()),
                mine_num: 1,
            },
            CellGroup {
                offsets: HashSet::from_iter([6, 7].into_iter()),
                mine_num: 1,
            },
            CellGroup {
                offsets: HashSet::from_iter([7, 2].into_iter()),
                mine_num: 1,
            },
            CellGroup {
                offsets: HashSet::from_iter([5, 6].into_iter()),
                mine_num: 1,
            },
            CellGroup {
                offsets: HashSet::from_iter([20, 200].into_iter()),
                mine_num: 1,
            },
        ];
        let output = merge_overlapping_groups(&groups);
        let mut set_extra = HashSet::default();
        set_extra.insert(&groups[5]);
        assert_eq!(
            output,
            vec![
                groups
                    .iter()
                    .filter(|x| x.offsets != HashSet::from_iter([20, 200].into_iter()))
                    .collect::<HashSet<_>>(),
                set_extra
            ]
        );
    }

    #[test]
    fn merge_overlapping_first_order_test() {
        let groups = vec![
            CellGroup {
                offsets: HashSet::from_iter([1, 2].into_iter()),
                ..CellGroup::default()
            }, // 0
            CellGroup {
                offsets: HashSet::from_iter([5, 4].into_iter()),
                ..CellGroup::default()
            }, // 1
            CellGroup {
                offsets: HashSet::from_iter([6, 7].into_iter()),
                ..CellGroup::default()
            }, // 2
            CellGroup {
                offsets: HashSet::from_iter([7, 2].into_iter()),
                ..CellGroup::default()
            }, // 3
            CellGroup {
                offsets: HashSet::from_iter([5, 6].into_iter()),
                ..CellGroup::default()
            }, // 4
            CellGroup {
                offsets: HashSet::from_iter([200, 20].into_iter()),
                ..CellGroup::default()
            }, // 5
        ];
        let output = merge_overlapping_groups_first_order(&groups);
        assert_eq!(
            output,
            vec![
                vec![&groups[0], &groups[3]],
                vec![&groups[1], &groups[4]],
                vec![&groups[2], &groups[3], &groups[4]],
                vec![&groups[3], &groups[0], &groups[2]],
                vec![&groups[4], &groups[1], &groups[2]],
                vec![&groups[5]],
            ]
        )
    }

    #[test]
    fn test_simulation_creation() {
        let sim = Simulation::new(4, 5, 6, CellCord(14, 7));
        assert_eq!(sim.board_cell_width, 4);
        assert_eq!(sim.board_cell_height, 5);
        assert_eq!(sim.state.iter().filter(|x| **x).count(), 6);
    }

    #[test]
    fn simulate_solve() {
        let mut game = Game::new_for_simulation(30, 16, 99, CellCord(14, 7));
        if LOGGING {
            println!("{:?}", &game.simulation);
        }
        match game.solve(CellCord(14, 7), true) {
            Ok(_) => (),
            Err(GameError::RevealedMine(_)) => (),
            Err(e) => panic!("{e}"),
        }
        if LOGGING {
            dbg!(game.state);
        };
    }

    // Test to figure out why infinite `Tried flagging a non flaggable at` issue
    #[test]
    fn simulate_infinite_flag_error() {
        let mut game = Game::new_for_simulation(5, 5, 7, CellCord(0, 0));
        game.simulation = Some(Simulation::new(5, 5, 7, CellCord(0, 0)));
        game.simulation.as_mut().unwrap().state =
            bitvec![0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,];
        match game.solve(CellCord(0, 0), true) {
            Ok(_) => (),
            Err(GameError::RevealedMine(_)) => (),
            Err(e) => panic!("{e}"),
        }
        game.save_state_info("test/FinalGameState.csv", true)
    }

    // Test calculates win rate in simulation.
    // Mainly works to test that many runs don't panic.
    #[test]
    fn simulate_win_rate_one_cell() {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let win_cnt = AtomicUsize::new(0);
        let lose_cnt = AtomicUsize::new(0);
        let unfinished_cnt = AtomicUsize::new(0);
        let initial_guess = CellCord(3, 3);
        // 10x faster on release so 10x more iterations can be done.
        let iteration_cnt = 1_000 * if cfg!(debug_assertions) { 1 } else { 10 };
        (0..iteration_cnt).into_par_iter().for_each(|_| {
            let mut game = Game::new_for_simulation(30, 16, 99, initial_guess);
            match game.solve(initial_guess, true) {
                Err(GameError::RevealedMine(_)) => lose_cnt.fetch_add(1, Ordering::Relaxed),
                Ok(_) => win_cnt.fetch_add(1, Ordering::Relaxed),
                Err(GameError::Unfinished) => unfinished_cnt.fetch_add(1, Ordering::Relaxed),
                Err(e) => panic!("{e}"),
            };
        });
        {
            let win_cnt = win_cnt.into_inner();
            let lose_cnt = lose_cnt.into_inner();
            let unfinished_cnt = unfinished_cnt.into_inner();
            println!("Win count: {}.", win_cnt);
            println!("Lose count: {}.", lose_cnt);
            println!("Unfinished count: {}.", unfinished_cnt);
            let winrate = win_cnt as f64 / (win_cnt + lose_cnt + unfinished_cnt) as f64;
            let winrate = winrate * 100f64;
            println!("Winrate: {winrate}");
        }
    }

    // Test calculates win rate in simulation per starting in each cell.
    #[test]
    #[ignore = "Takes very long to run"]
    fn simulate_win_rate_per_cell() {
        use rayon::prelude::*;
        use std::sync::Mutex;
        // 10x faster on release so 10x more iterations can be done.
        let iteration_cnt = 1_000 * if cfg!(debug_assertions) { 1 } else { 10 };
        let mut winrates = vec![];
        for i in 0..30 {
            winrates.push(vec![0.; 16]);
            for j in 0..16 {
                {
                    let win_cnt = Mutex::new(0);
                    let lose_cnt = Mutex::new(0);
                    let unfinished_cnt = Mutex::new(0);
                    let initial_guess = CellCord(i, j);
                    (0..iteration_cnt).into_par_iter().for_each(|_| {
                        let mut game = Game::new_for_simulation(30, 16, 99, initial_guess);
                        match game.solve(initial_guess, true) {
                            Err(GameError::RevealedMine(_)) => *lose_cnt.lock().unwrap() += 1,
                            Ok(_) => *win_cnt.lock().unwrap() += 1,
                            Err(GameError::Unfinished) => {
                                *unfinished_cnt.lock().unwrap() += 1;
                            }
                            Err(e) => panic!("{e}"),
                        }
                    });
                    {
                        println!("Initial guess {initial_guess:?}");
                        println!("Win count: {}.", win_cnt.lock().unwrap());
                        println!("Lose count: {}.", lose_cnt.lock().unwrap());
                        println!("Unfinished count: {}.", unfinished_cnt.lock().unwrap());
                        *lose_cnt.lock().unwrap() += *unfinished_cnt.lock().unwrap();
                        let win_cnt = *win_cnt.lock().unwrap();
                        let winrate = win_cnt as f64 / (win_cnt + *lose_cnt.lock().unwrap()) as f64;
                        let winrate = winrate * 100f64;
                        println!("Winrate: {winrate}");

                        winrates[i][j] = winrate;
                    }
                }
            }
        }
        println!("{winrates:?}");
    }

    // Test to figure out 'ERROR at self.cell_groups[5]=CellGroup { offsets: {}, mine_num: 1 } has more mines than cells to fill. Just removed 365' issue.
    #[test]
    #[rustfmt::skip]
    fn more_mines_than_cells_error() {
        let mut game = Game::new_for_simulation(30, 16, 99, CellCord(14, 7));
        game.simulation = Some(Simulation::new(30, 16, 99, CellCord(14, 7)));
        game.simulation.as_mut().unwrap().state = bitvec![
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
            1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,
            0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
            0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
            1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        match game.solve(CellCord(14, 7), true) {
            Ok(_) => (),
            Err(GameError::RevealedMine(c)) => println!("Revealed mine at {c:?}"),
            Err(GameError::IncorrectFlag) => println!("A flag was incorrect."),
            Err(e) => panic!("{e}"),
        }
        game.save_state_info("test/FinalGameState.csv", true)
    }

    // Test to figure out 'There were no valid combinations!' issue.
    #[test]
    #[rustfmt::skip]
    fn no_valid_combinations_error() {
        let mut game = Game::new_for_simulation(30, 16, 99, CellCord(14, 7));
        game.simulation = Some(Simulation::new(30, 16, 99, CellCord(14, 7)));
        game.simulation.as_mut().unwrap().state = bitvec![
            0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
            0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
            1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        match game.solve(CellCord(14, 7), true) {
            Ok(_) => (),
            Err(GameError::RevealedMine(c)) => println!("Revealed mine at {c:?}"),
            Err(GameError::IncorrectFlag) => println!("A flag was incorrect."),
            Err(e) => panic!("{e}"),
        }
        game.save_state_info("test/FinalGameState.csv", true)
    }
}
