use minesweeper_solver_in_rust::Game;

fn main(){
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

    let center = (game.cell_width/2, game.cell_height/2);
    // game.reveal(center);
    game.exit(false);
}
