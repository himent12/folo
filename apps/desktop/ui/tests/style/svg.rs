use folo_ui::{
    AssetSource, Bounds, DevicePixels, RenderSvgParams, SharedString, SvgRenderer, point, px, size,
};
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Two of the app's icons, counting loads (the renderer caches parsed trees).
#[derive(Default)]
struct Icons {
    loads: AtomicUsize,
}

impl AssetSource for Icons {
    fn load(&self, path: &str) -> anyhow::Result<Option<Cow<'static, [u8]>>> {
        self.loads.fetch_add(1, Ordering::Relaxed);
        Ok(match path {
            "icons/check.svg" => Some(Cow::Borrowed(include_bytes!("../../../assets/icons/check.svg"))),
            "icons/logo-frame.svg" => {
                Some(Cow::Borrowed(include_bytes!("../../../assets/icons/logo-frame.svg")))
            }
            _ => None,
        })
    }

    fn list(&self, _path: &str) -> anyhow::Result<Vec<SharedString>> {
        Ok(vec!["icons/check.svg".into(), "icons/logo-frame.svg".into()])
    }
}

#[test]
fn an_icon_at_16px_and_scale_1_5_is_a_48px_mask_with_coverage() {
    let icons = Arc::new(Icons::default());
    let renderer = SvgRenderer::new(icons.clone());
    // A 16 px icon on a 1.5x display: 24 device px, rendered at 2x.
    let bounds = Bounds::new(point(px(3.), px(7.)), size(px(16.), px(16.))).scale(1.5);
    let params = RenderSvgParams::for_bounds("icons/check.svg".into(), &bounds);
    assert_eq!(params.size, size(DevicePixels(48), DevicePixels(48)));

    let mask = renderer.render(&params).unwrap().expect("the icon exists");
    assert_eq!(mask.size, size(DevicePixels(48), DevicePixels(48)));
    assert_eq!(mask.data.len(), 48 * 48);
    let covered = mask.data.iter().filter(|a| **a > 0).count();
    assert!(covered > 48 && covered < 48 * 48 / 2, "a check mark's stroke, not empty or full: {covered}");
    assert!(mask.data.contains(&255), "the stroke's middle is fully covered");

    // Same tree from the cache at another size; a new mask id each time.
    let small = renderer
        .render(&RenderSvgParams { size: size(DevicePixels(32), DevicePixels(32)), ..params.clone() });
    let small = small.unwrap().unwrap();
    assert_eq!(small.size.width, DevicePixels(32));
    assert_ne!(small.id, mask.id);
    assert_eq!(icons.loads.load(Ordering::Relaxed), 1);
}

#[test]
fn missing_assets_and_zero_sizes() {
    let renderer = SvgRenderer::new(Arc::new(Icons::default()));
    let missing =
        RenderSvgParams { path: "icons/nope.svg".into(), size: size(DevicePixels(8), DevicePixels(8)) };
    assert!(renderer.render(&missing).unwrap().is_none());
    let zero =
        RenderSvgParams { path: "icons/check.svg".into(), size: size(DevicePixels(0), DevicePixels(8)) };
    assert!(renderer.render(&zero).is_err());
    let none = SvgRenderer::new(Arc::new(()));
    assert!(none.render(&missing).unwrap().is_none());

    // Another icon: the requested width decides the scale.
    let logo = RenderSvgParams {
        path: "icons/logo-frame.svg".into(),
        size: size(DevicePixels(40), DevicePixels(40)),
    };
    let mask = renderer.render(&logo).unwrap().unwrap();
    assert_eq!(mask.size.width, DevicePixels(40));
}
