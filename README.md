- This project solves minesweeper. Its a ported and significantly improved version of the python minesweeper solver I made previously.
- It's setup to solve minesweeper from `minesweeperonline.com/`, however, it can be setup to solve any minesweeper program as long as the .png files are replaced. It's expected that all images for tile/cell types are the same image size and correctly named.
    - If an unidentified tile/cell image is found it will be saved to `cell_images/Unidentified.png` before the program panics, so only the unexplored cell type image needs to be correct for the program to find the board and attempt solving. The other images can be set by renaming the generated `cell_images/Unidentified.png` when the computer finds a new cell type image.
    - It is possible the images in the repo won't work for you as the website has multiple possible tile images depending on computer monitor size.
- In `main.rs` there are some configuration options.
    - `NUMBER_OF_MINES` controls how many mines the solver expects there to be. This is `99` for a expert mode game.
- In `lib.rs` there are some configuration options.
    - `TIMEOUTS_ATTEMPTS_NUM` controls how many times the program is willing to rescan the page for an update. This exists because sometimes the board is screenshotted to be analyzed before the board updates.
    - `MAX_COMBINATIONS` controls how many combinations/permutations the probabilistic guess method is willing to go through before giving up and using a less accurate but quicker guess method instead.

![demo gif](/demo.gif)