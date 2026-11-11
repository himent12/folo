use folo_ui::{bounce, ease_in_out, ease_out_quint, linear, pulsating_between, quadratic};

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn easings_start_at_zero_and_end_at_one() {
    let quint = ease_out_quint();
    for f in [linear as fn(f32) -> f32, quadratic, ease_in_out] {
        assert!(close(f(0.), 0.) && close(f(1.), 1.));
    }
    assert!(close(quint(0.), 0.) && close(quint(1.), 1.));
    // GPUI's shapes in between.
    assert!(
        close(quadratic(0.5), 0.25) && close(ease_in_out(0.25), 0.125) && close(ease_in_out(0.75), 0.875)
    );
    assert!(close(quint(0.5), 1. - 0.5f32.powi(5)));
}

#[test]
fn bounce_and_pulse() {
    let b = bounce(linear);
    assert!(close(b(0.), 0.) && close(b(0.25), 0.5) && close(b(0.5), 1.) && close(b(1.), 0.));
    let p = pulsating_between(0.4, 1.0);
    // sin(0) = 0 -> the middle; sin(pi/2) = 1 -> max; sin(3pi/2) = -1 -> min.
    assert!(close(p(0.), 0.7) && close(p(0.25), 1.0) && close(p(0.75), 0.4));
}
