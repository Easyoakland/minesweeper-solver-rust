use minesweeper_solver_in_rust::{Game, CellCord};
// TODO in lib
// TODO save unidentified cell in correct location with name so it can easily be renamed to be added.
// Replace unwraps with unreachable! where appropriate.
// Convert conversions you can convert a `usize` to a `u32` and panic if the converted value doesn't fit: `.try_into().unwrap()`
// Convert return true or false for when a function does something to a Option where it makes sense. Like when it returns a tuple with true false and a value.

fn main(){
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

    game.solve(CellCord(0,0));
    game.save_state_info();
}
