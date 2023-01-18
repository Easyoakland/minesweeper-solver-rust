use minesweeper_solver_in_rust::{setup_capturer, Game};

fn main() {
    let mut capturer = setup_capturer(0);

    // TODO remove below underscores.
    let (_width, _height) = capturer.geometry();

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
        &mut capturer,
    );

    dbg!(game);
}
