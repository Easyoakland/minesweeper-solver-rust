use criterion::{/* black_box, */ criterion_group, criterion_main, Criterion};
use image::io;
use minesweeper_solver_in_rust::locate_all;

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
    c.bench_function("sub image search benchmark", |b| {
        b.iter(|| locate_all(&super_image, &sub_image))
    });
}

criterion_group!(benches, test_board_sub_image_search_benchmark);
criterion_main!(benches);
