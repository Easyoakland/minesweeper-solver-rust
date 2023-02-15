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

/* /// Expects buffer to be a RGBA image.
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
 */

/* /// Expects buffer to be a RGB image.
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
} */

// Below is more correct than function actually used, however, the number of combinations is too high and winrate actually goes down with correct partioning.
// Specifically this passes the following test while the one in use does not.
/*
       let groups = vec![
           CellGroup {
               offsets: HashSet::from([248, 249]),
               mine_num: 1,
           },
           CellGroup {
               offsets: HashSet::from([278, 248]),
               mine_num: 1,
           },
           CellGroup {
               offsets: HashSet::from([279, 249]),
               mine_num: 1,
           },
           CellGroup {
               offsets: HashSet::from([279, 278]),
               mine_num: 1,
           },
       ];
       let output = merge_overlapping_groups(&groups);
       assert_eq!(output, vec![groups.iter().collect::<HashSet<_>>()]);

       let groups = vec![
           CellGroup {
               offsets: HashSet::from([1, 2]),
               mine_num: 1,
           },
           CellGroup {
               offsets: HashSet::from([5, 4]),
               mine_num: 1,
           },
           CellGroup {
               offsets: HashSet::from([6, 7]),
               mine_num: 1,
           },
           CellGroup {
               offsets: HashSet::from([7, 2]),
               mine_num: 1,
           },
           CellGroup {
               offsets: HashSet::from([5, 6]),
               mine_num: 1,
           },
           CellGroup {
               offsets: HashSet::from([20, 200]),
               mine_num: 1,
           },
       ];
       let output = merge_overlapping_groups(&groups);
       dbg!(&output);
       let mut set_extra = HashSet::new();
       set_extra.insert(&groups[5]);
       assert_eq!(
           output,
           vec![
               groups
                   .iter()
                   .filter(|x| x.offsets != HashSet::from([20, 200]))
                   .collect::<HashSet<_>>(),
               set_extra
           ]
       );
*/
/*
/// For each Subgroup if input element intersects then merge element into `Subgroup` and mark the `Subgroup`.
/// Then remove all marked Subgroup and combine before adding back.
/// If doesn't intersect make new Subgroup.
fn merge_overlapping_groups<'a>(input: &'a [CellGroup]) -> Vec<HashSet<&'a CellGroup>> {
    #[derive(Debug, Default)]
    struct Subgroup {
        offsets: HashSet<usize>,
        sets: HashSet<usize>,
    }

    let mut state: Vec<Subgroup> = Vec::new();
    for (i, e) in input.into_iter().enumerate() {
        let mut marks = vec![];
        let mut temp = vec![Subgroup {
            offsets: e.offsets.clone(),
            sets: {
                let mut set = HashSet::new();
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
        let mut set = HashSet::new();
        for set_idx in subgroup.sets {
            set.insert(&input[set_idx]);
            // set.insert(set_idx); // inserts indexes in input instead of values
        }
        out.push(set);
    }
    out
}
 */
