- This project solves minesweeper.
- It has a `42%` win-rate.
- It can simulate a full game in `1 ms` on a single core and can run a simulation on each core simultaneously. Caching, avoiding heap allocations, and other optimizations are used to bring the time this low.
- The speed of the solver when playing a real game using screen capture is entirely dependent on how fast your computer can capture frames as they render. Multiple actions are queued at a time so less screen captures are needed.
- The algorithm works as follows:
    1.  First solve all obvious deterministic methods.
    2. Next try local solving in which mines are placed in possible positions for a numbered cell, and it is determined if any mine is obviously wrong or right.
    3. Then use a probabilistic guess algorithm to enumerate every possible position of mines on the board of visible cells and choose the least likely position to be a mine.
    4. When the number of cells and mines left is small enough to not go over the max combinations threshold, include all mines left and all cells on the board (not just those that border the explored edge) to increase odds of winning in the endgame by analyzing every possible placement of all remaining mines.
- It's setup to solve minesweeper from `minesweeperonline.com/`, however, it can be setup to solve any minesweeper program as long as the .png files are replaced. It's expected that all images for tile/cell types are the same image size and correctly named.
    - If an unidentified tile/cell image is found it will be saved to `cell_images/Unidentified.png` before the program panics, so only the unexplored cell type image needs to be correct for the program to find the board and attempt solving. The other images can be set by renaming the generated `cell_images/Unidentified.png` when the computer finds a new cell type image.
    - It is possible the images in the repo won't work for you as the website has multiple possible tile images depending on computer monitor size.
- In `main.rs` there are some configuration options.
    - `NUMBER_OF_MINES` controls how many mines the solver expects there to be. This is `99` for an expert mode game.
- In `lib.rs` there are some configuration options.
    - `TIMEOUTS_ATTEMPTS_NUM` controls how many times the program is willing to re-scan the page for an update. This exists because sometimes the board is screenshotted to be analyzed before the board updates.
    - `MAX_COMBINATIONS` controls how many combinations the probabilistic guess method is willing to go through before giving up and using a less accurate but quicker guess method instead.

![demo gif](/demo.gif)