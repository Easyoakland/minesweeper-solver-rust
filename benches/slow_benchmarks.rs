use criterion::{black_box, criterion_group, criterion_main, Criterion};

use minesweeper_solver_in_rust::{capture_frame, setup_capturer};
use std::time::Duration;
fn capture_frame_benchmark(c: &mut Criterion) {
    let mut capturer = setup_capturer(0);
    let one_second = Duration::new(1, 0);
    let one_frame = one_second / 60;

    let (width, height) = (capturer.width(), capturer.height());
    c.bench_function("capture frame", |b| {
        b.iter(|| capture_frame(black_box(&mut capturer), one_frame, width, height))
    });
}

fn raw_frame_benchmark(c: &mut Criterion) {
    let mut capturer = setup_capturer(0);
    c.bench_function("raw frame", |b| {
        b.iter(|| {
            let _temp = black_box(capturer.frame());
            ()
        });
    });
}

fn raw_valid_frame_benchmark(c: &mut Criterion) {
    let mut capturer = setup_capturer(0);
    c.bench_function("raw valid frame", |b| {
        b.iter(|| loop {
            let temp = black_box(capturer.frame());
            match temp {
                Ok(_) => return (),
                Err(_) => continue,
            }
        });
    });
}

// use std::thread;
fn raw_valid_frame_nonblack_benchmark(c: &mut Criterion) {
    let mut capturer = setup_capturer(0);
    c.bench_function("raw valid frame without all black", |b| {
        b.iter(|| loop {
            let temp = capturer.frame();
            match temp {
                Ok(buffer) => {
                    // Make sure the image is not a failed black screen.
                    if !buffer.to_vec().iter().any(|&x| x != 0) {
                        // thread::sleep(Duration::new(0, 1)); // sleep 1ms
                        continue;
                    };
                    return ();
                }
                Err(_) => continue,
            }
        });
    });
}

criterion_group! {

    name = benches2;
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(5));
    targets = capture_frame_benchmark, raw_frame_benchmark, raw_valid_frame_benchmark, raw_valid_frame_nonblack_benchmark
}

criterion_main!(benches2);
