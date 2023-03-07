use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::io;
use minesweeper_solver_in_rust::{
    locate_all, read_image, setup_capturer, CellCord, Game, GameError,
};
use rand::{seq::SliceRandom, thread_rng};
use std::collections::HashSet;

/* fn fibonacci(n: u64) -> u64 {
    /*     match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n-1) + fibonacci(n-2),
    } */
    let mut a = 0;
    let mut b = 1;

    match n {
        0 => b,
        _ => {
            for _ in 0..n {
                let c = a + b;
                a = b;
                b = c;
            }
            b
        }
    }
}

fn fibonacci_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
} */

fn test_board_sub_image_search_benchmark(c: &mut Criterion) {
    let super_image = io::Reader::open("test_in/subimage_search/1920x1080_board.png")
        .expect("Couldn't read super image.")
        .decode()
        .expect("Unsupported Type");
    let sub_image = io::Reader::open("test_in/subimage_search/cell.png")
        .expect("Couldn't read sub image")
        .decode()
        .expect("Unsupported Type");
    c.bench_function("sub image search benchmark using locate all", |b| {
        b.iter(|| locate_all(&super_image, &sub_image))
    });
}

fn test_identify_cell_benchmark(c: &mut Criterion) {
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
    c.bench_function("identify cell on unexplored", |b| {
        b.iter(|| Game::identify_cell_benchmark_pub_func(black_box(&mut game)))
    });
}

fn sort_binary_or_linear_or_hashmap(c: &mut Criterion) {
    let list1 = HashSet::from([1, 2, 3, 4, 5, 6, 7, 200, 12, 10, 12, 13, 14, 15, 16]);
    let mut list2 = [7, 6, 5, 4, 3, 2, 1, 99, 88];
    fn linear_test(list1: &HashSet<i32>, list2: [i32; 9]) -> i32 {
        let mut count = 0;
        for item in list1 {
            if list2.contains(&item) {
                count += 1;
            }
        }
        count
    }
    fn binary_test(list1: &HashSet<i32>, list2: [i32; 9]) -> i32 {
        let mut count = 0;
        for item in list1 {
            if let Ok(_) = list2.binary_search(&item) {
                count += 1;
            }
        }
        count
    }
    fn hashmap_test(list1: &HashSet<i32>, list2: [i32; 9]) -> i32 {
        let mut count = 0;
        let list2 = HashSet::from(list2);
        for item in list1 {
            if list2.contains(&item) {
                count += 1;
            }
        }
        count
    }
    let mut rng = thread_rng();
    c.bench_function("linear itersection count", |b| {
        list2.shuffle(&mut rng);
        b.iter(|| linear_test(&list1, list2));
    });
    c.bench_function("binary sort itersection count", |b| {
        list2.shuffle(&mut rng);
        list2.sort();
        b.iter(|| binary_test(&list1, list2));
    });
    c.bench_function("Hashmap itersection count", |b| {
        list2.shuffle(&mut rng);
        b.iter(|| hashmap_test(&list1, list2));
    });
}

fn simulate_win_rate_one_cell_benchmark(c: &mut Criterion) {
    fn simulate_win_rate_one_cell_one_time() {
        use std::sync::Mutex;
        let win_cnt = Mutex::new(0);
        let lose_cnt = Mutex::new(0);
        let unfinished_cnt = Mutex::new(0);
        let initial_guess = CellCord(2, 3);
        let mut game = Game::new_for_simulation(30, 16, 99, initial_guess);
        match game.solve(initial_guess, true) {
            Err(GameError::RevealedMine(_)) => *lose_cnt.lock().unwrap() += 1,
            Ok(_) => *win_cnt.lock().unwrap() += 1,
            Err(GameError::Unfinished) => {
                *unfinished_cnt.lock().unwrap() += 1;
            }
            Err(e) => panic!("{e}"),
        };
    }
    c.bench_function("simulate one run", |b| {
        b.iter(|| simulate_win_rate_one_cell_one_time());
    });
}

criterion_group!(
    benches,
    test_board_sub_image_search_benchmark,
    test_identify_cell_benchmark,
    sort_binary_or_linear_or_hashmap,
    simulate_win_rate_one_cell_benchmark
);
criterion_main!(benches);
