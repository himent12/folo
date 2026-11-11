//! Picking the face of a family that best matches a requested weight, style and stretch: the CSS Fonts
//! level 3 matching algorithm (§5.2), which is what GPUI's Linux backend gets from font-kit's
//! `find_best_match`.

use crate::{FontStyle, FontWeight};

/// Font stretch on the CSS scale (100 = normal, 50 = ultra-condensed, 200 = ultra-expanded).
pub(super) type Stretch = f32;

/// The properties of one face, or of the request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct FaceProperties {
    pub weight: FontWeight,
    pub style: FontStyle,
    pub stretch: Stretch,
}

/// The index of the best match for `query` among `candidates`, `None` when there are none.
pub(super) fn find_best_match(candidates: &[FaceProperties], query: &FaceProperties) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }
    let mut matching: Vec<usize> = (0..candidates.len()).collect();

    // Stretch: exact, else narrower first when the request is normal or narrower, wider first otherwise.
    let stretch = closest(&matching, candidates, query.stretch, |c| c.stretch, query.stretch <= 100.);
    matching.retain(|&i| candidates[i].stretch == stretch);

    // Style: italic falls back to oblique then normal; oblique to italic then normal; normal to oblique
    // then italic.
    let order: [FontStyle; 3] = match query.style {
        FontStyle::Italic => [FontStyle::Italic, FontStyle::Oblique, FontStyle::Normal],
        FontStyle::Oblique => [FontStyle::Oblique, FontStyle::Italic, FontStyle::Normal],
        FontStyle::Normal => [FontStyle::Normal, FontStyle::Oblique, FontStyle::Italic],
    };
    let style = order.into_iter().find(|s| matching.iter().any(|&i| candidates[i].style == *s))?;
    matching.retain(|&i| candidates[i].style == style);

    // Weight: 400..=500 looks up to 500, then down, then above 500; below 400 looks down then up; above
    // 500 looks up then down.
    let want = query.weight.0;
    let weight_of = |i: usize| candidates[i].weight.0;
    if let Some(&exact) = matching.iter().find(|&&i| weight_of(i) == want) {
        return Some(exact);
    }
    let pick_min = |pred: &dyn Fn(f32) -> bool| {
        matching
            .iter()
            .copied()
            .filter(|&i| pred(weight_of(i)))
            .min_by(|&a, &b| weight_of(a).total_cmp(&weight_of(b)))
    };
    let pick_max = |pred: &dyn Fn(f32) -> bool| {
        matching
            .iter()
            .copied()
            .filter(|&i| pred(weight_of(i)))
            .max_by(|&a, &b| weight_of(a).total_cmp(&weight_of(b)))
    };
    if (400. ..=500.).contains(&want) {
        pick_min(&|w| w > want && w <= 500.)
            .or_else(|| pick_max(&|w| w < want))
            .or_else(|| pick_min(&|w| w > 500.))
    } else if want < 400. {
        pick_max(&|w| w < want).or_else(|| pick_min(&|w| w > want))
    } else {
        pick_min(&|w| w > want).or_else(|| pick_max(&|w| w < want))
    }
}

/// The value of `key` among `indices` closest to `want` by the CSS stretch rule.
fn closest(
    indices: &[usize],
    candidates: &[FaceProperties],
    want: f32,
    key: impl Fn(&FaceProperties) -> f32,
    narrower_first: bool,
) -> f32 {
    let values = indices.iter().map(|&i| key(&candidates[i]));
    if values.clone().any(|v| v == want) {
        return want;
    }
    let below = values.clone().filter(|&v| v < want).max_by(f32::total_cmp);
    let above = values.filter(|&v| v > want).min_by(f32::total_cmp);
    let (first, second) = if narrower_first { (below, above) } else { (above, below) };
    first.or(second).unwrap_or(want)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(weight: f32, style: FontStyle) -> FaceProperties {
        FaceProperties { weight: FontWeight(weight), style, stretch: 100. }
    }

    #[test]
    fn weights_follow_the_css_rule() {
        let geist =
            [face(400., FontStyle::Normal), face(500., FontStyle::Normal), face(600., FontStyle::Normal)];
        let pick = |w: f32| find_best_match(&geist, &face(w, FontStyle::Normal)).map(|i| geist[i].weight.0);
        assert_eq!(pick(400.), Some(400.));
        assert_eq!(pick(500.), Some(500.));
        assert_eq!(pick(600.), Some(600.));
        // 450 looks up to 500 first.
        assert_eq!(pick(450.), Some(500.));
        // Bold has no exact face: heavier first, then the heaviest lighter one.
        assert_eq!(pick(700.), Some(600.));
        // Light looks lighter first, then the lightest heavier one.
        assert_eq!(pick(300.), Some(400.));
        assert_eq!(pick(550.), Some(600.));

        let sparse = [face(300., FontStyle::Normal), face(700., FontStyle::Normal)];
        let pick = |w: f32| find_best_match(&sparse, &face(w, FontStyle::Normal)).map(|i| sparse[i].weight.0);
        // 400 has nothing in 400..=500, so it goes down before going above 500.
        assert_eq!(pick(400.), Some(300.));
        assert_eq!(pick(500.), Some(300.));
    }

    #[test]
    fn style_falls_back_in_order() {
        let faces = [face(400., FontStyle::Normal), face(400., FontStyle::Oblique)];
        assert_eq!(find_best_match(&faces, &face(400., FontStyle::Italic)), Some(1));
        assert_eq!(find_best_match(&faces, &face(400., FontStyle::Normal)), Some(0));
        let upright = [face(400., FontStyle::Normal)];
        assert_eq!(find_best_match(&upright, &face(700., FontStyle::Italic)), Some(0));
        assert_eq!(find_best_match(&[], &face(400., FontStyle::Normal)), None);
    }

    #[test]
    fn stretch_prefers_narrower_for_normal_requests() {
        let mut condensed = face(400., FontStyle::Normal);
        condensed.stretch = 75.;
        let mut expanded = face(400., FontStyle::Normal);
        expanded.stretch = 125.;
        assert_eq!(find_best_match(&[expanded, condensed], &face(400., FontStyle::Normal)), Some(1));
        let mut wide = face(400., FontStyle::Normal);
        wide.stretch = 150.;
        assert_eq!(find_best_match(&[condensed, expanded], &wide), Some(1));
    }
}
