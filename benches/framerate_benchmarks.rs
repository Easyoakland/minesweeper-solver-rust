use captrs::{Capturer, Bgr8};
use criterion::{/* black_box,  */criterion_group, criterion_main, Criterion};
use std::time::Duration;

/* use minesweeper_solver_in_rust::{capture_frame_scrap, setup_capturer};
fn capture_frame_benchmark(c: &mut Criterion) {
    let mut capturer = setup_capturer(0);
    let one_second = Duration::new(1, 0);
    let one_frame = one_second / 60;

    let (width, height) = (capturer.width(), capturer.height());
    c.bench_function("capture frame", |b| {
        b.iter(|| capture_frame_scrap(black_box(&mut capturer), one_frame, width, height))
    });
} */

/* fn raw_frame_benchmark(c: &mut Criterion) {
    let mut capturer = setup_capturer(0);
    c.bench_function("raw frame", |b| {
        b.iter(|| {
            let _temp = black_box(capturer.frame());
            ()
        });
    });
} */

/* fn raw_valid_frame_benchmark(c: &mut Criterion) {
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
} */

/* // use std::thread;
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
} */

fn raw_valid_frame_nonblack_benchmark_captrs(c: &mut Criterion) {
    let mut capturer = Capturer::new(0).unwrap();
    c.bench_function("raw valid frame without all black using captrs", |b| {
        b.iter(|| loop {
            let temp = capturer.capture_frame();
            match temp {
                Ok(ps) => {

                    let mut rgb_vec = Vec::new();
                    for Bgr8 { r, g, b, .. } in ps.into_iter() {
                        rgb_vec.push(r);
                        rgb_vec.push(g);
                        rgb_vec.push(b);
                    }

                    // Make sure the image is not a failed black screen.
                    if !rgb_vec.iter().any(|&x| x != 0) {
                        // thread::sleep(Duration::new(0, 1)); // sleep 1ms
                        println!("All black");
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

    name = framerate_benchmarks;
    config = Criterion::default().sample_size(60).measurement_time(Duration::from_secs(30));
    targets = /* capture_frame_benchmark, raw_frame_benchmark, raw_valid_frame_benchmark, raw_valid_frame_nonblack_benchmark, */ raw_valid_frame_nonblack_benchmark_captrs
}

criterion_main!(framerate_benchmarks);
