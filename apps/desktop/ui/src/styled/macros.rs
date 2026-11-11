// Adapted from GPUI (Apache-2.0): crates/gpui_macros/src/styles.rs
//! The method generators behind `Styled` (GPUI generates the same methods with proc macros). Each
//! `*_methods!` macro expands, inside the trait, to one family: a setter taking a value, and setters
//! with a fixed value (Tailwind names: `_0`, `_px`, `_full`, `_auto`, the border and radius scales).

/// Setters taking a value: `name(length)`, setting every listed field (GPUI's `generate_custom_value_setter`).
macro_rules! value_setters {
    ($( $(#[$doc:meta])* $name:ident: $ty:ident => [$($($field:ident).+),+]; )*) => {$(
        $(#[$doc])*
        fn $name(mut self, length: impl Clone + Into<$ty>) -> Self {
            let style = self.style();
            $( style.$($field).+ = Some(length.clone().into()); )+
            self
        }
    )*};
}

/// Setters with fixed values, grouped by the fields they set (GPUI's `generate_predefined_setter`).
macro_rules! fixed_setters {
    ($( $fields:tt { $($name:ident = $value:expr),* $(,)? } )*) => {
        $( $( fixed_setters!(@one $fields $name = $value); )* )*
    };
    (@one [$($($field:ident).+),+] $name:ident = $value:expr) => {
        #[doc = concat!("Sets `", stringify!($($($field).+),+), "` to `", stringify!($value), "`.")]
        fn $name(mut self) -> Self {
            let style = self.style();
            $( style.$($field).+ = Some(($value).into()); )+
            self
        }
    };
}

/// Margins: `m`, `mt`, `mb`, `my`, `mx`, `ml`, `mr` (`Length`, so `auto` too).
macro_rules! margin_methods {
    () => {
        value_setters! {
            /// Sets the margin on every side.
            m: Length => [margin.top, margin.bottom, margin.left, margin.right];
            /// Sets the top margin.
            mt: Length => [margin.top];
            /// Sets the bottom margin.
            mb: Length => [margin.bottom];
            /// Sets the top and bottom margins.
            my: Length => [margin.top, margin.bottom];
            /// Sets the left and right margins.
            mx: Length => [margin.left, margin.right];
            /// Sets the left margin.
            ml: Length => [margin.left];
            /// Sets the right margin.
            mr: Length => [margin.right];
        }
        fixed_setters! {
            [margin.top, margin.bottom, margin.left, margin.right]
                { m_0 = px(0.), m_px = px(1.), m_full = relative(1.), m_auto = auto() }
            [margin.top] { mt_0 = px(0.), mt_px = px(1.), mt_full = relative(1.), mt_auto = auto() }
            [margin.bottom] { mb_0 = px(0.), mb_px = px(1.), mb_full = relative(1.), mb_auto = auto() }
            [margin.top, margin.bottom] { my_0 = px(0.), my_px = px(1.), my_full = relative(1.), my_auto = auto() }
            [margin.left, margin.right] { mx_0 = px(0.), mx_px = px(1.), mx_full = relative(1.), mx_auto = auto() }
            [margin.left] { ml_0 = px(0.), ml_px = px(1.), ml_full = relative(1.), ml_auto = auto() }
            [margin.right] { mr_0 = px(0.), mr_px = px(1.), mr_full = relative(1.), mr_auto = auto() }
        }
    };
}

/// Padding: `p`, `pt`, `pb`, `px`, `py`, `pl`, `pr` (`DefiniteLength`: no `auto`).
macro_rules! padding_methods {
    () => {
        value_setters! {
            /// Sets the padding on every side.
            p: DefiniteLength => [padding.top, padding.bottom, padding.left, padding.right];
            /// Sets the top padding.
            pt: DefiniteLength => [padding.top];
            /// Sets the bottom padding.
            pb: DefiniteLength => [padding.bottom];
            /// Sets the left and right padding.
            px: DefiniteLength => [padding.left, padding.right];
            /// Sets the top and bottom padding.
            py: DefiniteLength => [padding.top, padding.bottom];
            /// Sets the left padding.
            pl: DefiniteLength => [padding.left];
            /// Sets the right padding.
            pr: DefiniteLength => [padding.right];
        }
        fixed_setters! {
            [padding.top, padding.bottom, padding.left, padding.right]
                { p_0 = px(0.), p_px = px(1.), p_full = relative(1.) }
            [padding.top] { pt_0 = px(0.), pt_px = px(1.), pt_full = relative(1.) }
            [padding.bottom] { pb_0 = px(0.), pb_px = px(1.), pb_full = relative(1.) }
            [padding.left, padding.right] { px_0 = px(0.), px_px = px(1.), px_full = relative(1.) }
            [padding.top, padding.bottom] { py_0 = px(0.), py_px = px(1.), py_full = relative(1.) }
            [padding.left] { pl_0 = px(0.), pl_px = px(1.), pl_full = relative(1.) }
            [padding.right] { pr_0 = px(0.), pr_px = px(1.), pr_full = relative(1.) }
        }
    };
}

/// Insets: `inset`, `top`, `bottom`, `left`, `right` (plus `relative` / `absolute` in the trait).
macro_rules! position_methods {
    () => {
        value_setters! {
            /// Sets the inset on every side.
            inset: Length => [inset.top, inset.right, inset.bottom, inset.left];
            /// Sets the top inset.
            top: Length => [inset.top];
            /// Sets the bottom inset.
            bottom: Length => [inset.bottom];
            /// Sets the left inset.
            left: Length => [inset.left];
            /// Sets the right inset.
            right: Length => [inset.right];
        }
        fixed_setters! {
            [inset.top, inset.right, inset.bottom, inset.left]
                { inset_0 = px(0.), inset_px = px(1.), inset_full = relative(1.), inset_auto = auto() }
            [inset.top] { top_0 = px(0.), top_px = px(1.), top_full = relative(1.), top_auto = auto() }
            [inset.bottom] { bottom_0 = px(0.), bottom_px = px(1.), bottom_full = relative(1.), bottom_auto = auto() }
            [inset.left] { left_0 = px(0.), left_px = px(1.), left_full = relative(1.), left_auto = auto() }
            [inset.right] { right_0 = px(0.), right_px = px(1.), right_full = relative(1.), right_auto = auto() }
        }
    };
}

/// Sizes and gaps: `w`, `h`, `size`, `min_w`, `min_h`, `max_w`, `max_h`, `gap`, `gap_x`, `gap_y`.
macro_rules! size_methods {
    () => {
        value_setters! {
            /// Sets the width.
            w: Length => [size.width];
            /// Sets the height.
            h: Length => [size.height];
            /// Sets the width and the height.
            size: Length => [size.width, size.height];
            /// Sets the minimum width.
            min_w: Length => [min_size.width];
            /// Sets the minimum height.
            min_h: Length => [min_size.height];
            /// Sets the maximum width.
            max_w: Length => [max_size.width];
            /// Sets the maximum height.
            max_h: Length => [max_size.height];
            /// Sets the gap between children on both axes.
            gap: DefiniteLength => [gap.width, gap.height];
            /// Sets the gap between columns.
            gap_x: DefiniteLength => [gap.width];
            /// Sets the gap between rows.
            gap_y: DefiniteLength => [gap.height];
        }
        fixed_setters! {
            [size.width] { w_0 = px(0.), w_px = px(1.), w_full = relative(1.), w_auto = auto() }
            [size.height] { h_0 = px(0.), h_px = px(1.), h_full = relative(1.), h_auto = auto() }
            [size.width, size.height]
                { size_0 = px(0.), size_px = px(1.), size_full = relative(1.), size_auto = auto() }
            [min_size.width] { min_w_0 = px(0.), min_w_px = px(1.), min_w_full = relative(1.), min_w_auto = auto() }
            [min_size.height] { min_h_0 = px(0.), min_h_px = px(1.), min_h_full = relative(1.), min_h_auto = auto() }
            [max_size.width] { max_w_0 = px(0.), max_w_px = px(1.), max_w_full = relative(1.), max_w_auto = auto() }
            [max_size.height] { max_h_0 = px(0.), max_h_px = px(1.), max_h_full = relative(1.), max_h_auto = auto() }
            [gap.width, gap.height] { gap_0 = px(0.), gap_px = px(1.), gap_full = relative(1.) }
            [gap.width] { gap_x_0 = px(0.), gap_x_px = px(1.), gap_x_full = relative(1.) }
            [gap.height] { gap_y_0 = px(0.), gap_y_px = px(1.), gap_y_full = relative(1.) }
        }
    };
}

/// Corner radii: `rounded`, `rounded_{t,b,r,l,tl,tr,bl,br}`, each with GPUI's radius scale.
macro_rules! rounded_methods {
    () => {
        value_setters! {
            /// Sets every corner radius.
            rounded: AbsoluteLength => [corner_radii.top_left, corner_radii.top_right, corner_radii.bottom_right, corner_radii.bottom_left];
            /// Sets the top corner radii.
            rounded_t: AbsoluteLength => [corner_radii.top_left, corner_radii.top_right];
            /// Sets the bottom corner radii.
            rounded_b: AbsoluteLength => [corner_radii.bottom_left, corner_radii.bottom_right];
            /// Sets the right corner radii.
            rounded_r: AbsoluteLength => [corner_radii.top_right, corner_radii.bottom_right];
            /// Sets the left corner radii.
            rounded_l: AbsoluteLength => [corner_radii.top_left, corner_radii.bottom_left];
            /// Sets the top-left corner radius.
            rounded_tl: AbsoluteLength => [corner_radii.top_left];
            /// Sets the top-right corner radius.
            rounded_tr: AbsoluteLength => [corner_radii.top_right];
            /// Sets the bottom-left corner radius.
            rounded_bl: AbsoluteLength => [corner_radii.bottom_left];
            /// Sets the bottom-right corner radius.
            rounded_br: AbsoluteLength => [corner_radii.bottom_right];
        }
        fixed_setters! {
            [corner_radii.top_left, corner_radii.top_right, corner_radii.bottom_right, corner_radii.bottom_left] {
                rounded_none = px(0.), rounded_xs = rems(0.125), rounded_sm = rems(0.25), rounded_md = rems(0.375),
                rounded_lg = rems(0.5), rounded_xl = rems(0.75), rounded_2xl = rems(1.), rounded_3xl = rems(1.5),
                rounded_full = px(9999.),
            }
            [corner_radii.top_left, corner_radii.top_right] {
                rounded_t_none = px(0.), rounded_t_xs = rems(0.125), rounded_t_sm = rems(0.25),
                rounded_t_md = rems(0.375), rounded_t_lg = rems(0.5), rounded_t_xl = rems(0.75),
                rounded_t_2xl = rems(1.), rounded_t_3xl = rems(1.5), rounded_t_full = px(9999.),
            }
            [corner_radii.bottom_left, corner_radii.bottom_right] {
                rounded_b_none = px(0.), rounded_b_xs = rems(0.125), rounded_b_sm = rems(0.25),
                rounded_b_md = rems(0.375), rounded_b_lg = rems(0.5), rounded_b_xl = rems(0.75),
                rounded_b_2xl = rems(1.), rounded_b_3xl = rems(1.5), rounded_b_full = px(9999.),
            }
            [corner_radii.top_right, corner_radii.bottom_right] {
                rounded_r_none = px(0.), rounded_r_xs = rems(0.125), rounded_r_sm = rems(0.25),
                rounded_r_md = rems(0.375), rounded_r_lg = rems(0.5), rounded_r_xl = rems(0.75),
                rounded_r_2xl = rems(1.), rounded_r_3xl = rems(1.5), rounded_r_full = px(9999.),
            }
            [corner_radii.top_left, corner_radii.bottom_left] {
                rounded_l_none = px(0.), rounded_l_xs = rems(0.125), rounded_l_sm = rems(0.25),
                rounded_l_md = rems(0.375), rounded_l_lg = rems(0.5), rounded_l_xl = rems(0.75),
                rounded_l_2xl = rems(1.), rounded_l_3xl = rems(1.5), rounded_l_full = px(9999.),
            }
            [corner_radii.top_left] {
                rounded_tl_none = px(0.), rounded_tl_xs = rems(0.125), rounded_tl_sm = rems(0.25),
                rounded_tl_md = rems(0.375), rounded_tl_lg = rems(0.5), rounded_tl_xl = rems(0.75),
                rounded_tl_2xl = rems(1.), rounded_tl_3xl = rems(1.5), rounded_tl_full = px(9999.),
            }
            [corner_radii.top_right] {
                rounded_tr_none = px(0.), rounded_tr_xs = rems(0.125), rounded_tr_sm = rems(0.25),
                rounded_tr_md = rems(0.375), rounded_tr_lg = rems(0.5), rounded_tr_xl = rems(0.75),
                rounded_tr_2xl = rems(1.), rounded_tr_3xl = rems(1.5), rounded_tr_full = px(9999.),
            }
            [corner_radii.bottom_left] {
                rounded_bl_none = px(0.), rounded_bl_xs = rems(0.125), rounded_bl_sm = rems(0.25),
                rounded_bl_md = rems(0.375), rounded_bl_lg = rems(0.5), rounded_bl_xl = rems(0.75),
                rounded_bl_2xl = rems(1.), rounded_bl_3xl = rems(1.5), rounded_bl_full = px(9999.),
            }
            [corner_radii.bottom_right] {
                rounded_br_none = px(0.), rounded_br_xs = rems(0.125), rounded_br_sm = rems(0.25),
                rounded_br_md = rems(0.375), rounded_br_lg = rems(0.5), rounded_br_xl = rems(0.75),
                rounded_br_2xl = rems(1.), rounded_br_3xl = rems(1.5), rounded_br_full = px(9999.),
            }
        }
    };
}

/// Border widths: `border`, `border_{t,b,r,l,x,y}`, each with GPUI's pixel scale (0 to 32).
macro_rules! border_methods {
    () => {
        value_setters! {
            /// Sets the border width on every side.
            border: AbsoluteLength => [border_widths.top, border_widths.right, border_widths.bottom, border_widths.left];
            /// Sets the top border width.
            border_t: AbsoluteLength => [border_widths.top];
            /// Sets the bottom border width.
            border_b: AbsoluteLength => [border_widths.bottom];
            /// Sets the right border width.
            border_r: AbsoluteLength => [border_widths.right];
            /// Sets the left border width.
            border_l: AbsoluteLength => [border_widths.left];
            /// Sets the left and right border widths.
            border_x: AbsoluteLength => [border_widths.left, border_widths.right];
            /// Sets the top and bottom border widths.
            border_y: AbsoluteLength => [border_widths.top, border_widths.bottom];
        }
        border_scale! { [border_widths.top, border_widths.right, border_widths.bottom, border_widths.left]
            border_0 border_1 border_2 border_3 border_4 border_5 border_6 border_7 border_8 border_9
            border_10 border_11 border_12 border_16 border_20 border_24 border_32 }
        border_scale! { [border_widths.top]
            border_t_0 border_t_1 border_t_2 border_t_3 border_t_4 border_t_5 border_t_6 border_t_7 border_t_8
            border_t_9 border_t_10 border_t_11 border_t_12 border_t_16 border_t_20 border_t_24 border_t_32 }
        border_scale! { [border_widths.bottom]
            border_b_0 border_b_1 border_b_2 border_b_3 border_b_4 border_b_5 border_b_6 border_b_7 border_b_8
            border_b_9 border_b_10 border_b_11 border_b_12 border_b_16 border_b_20 border_b_24 border_b_32 }
        border_scale! { [border_widths.right]
            border_r_0 border_r_1 border_r_2 border_r_3 border_r_4 border_r_5 border_r_6 border_r_7 border_r_8
            border_r_9 border_r_10 border_r_11 border_r_12 border_r_16 border_r_20 border_r_24 border_r_32 }
        border_scale! { [border_widths.left]
            border_l_0 border_l_1 border_l_2 border_l_3 border_l_4 border_l_5 border_l_6 border_l_7 border_l_8
            border_l_9 border_l_10 border_l_11 border_l_12 border_l_16 border_l_20 border_l_24 border_l_32 }
        border_scale! { [border_widths.left, border_widths.right]
            border_x_0 border_x_1 border_x_2 border_x_3 border_x_4 border_x_5 border_x_6 border_x_7 border_x_8
            border_x_9 border_x_10 border_x_11 border_x_12 border_x_16 border_x_20 border_x_24 border_x_32 }
        border_scale! { [border_widths.top, border_widths.bottom]
            border_y_0 border_y_1 border_y_2 border_y_3 border_y_4 border_y_5 border_y_6 border_y_7 border_y_8
            border_y_9 border_y_10 border_y_11 border_y_12 border_y_16 border_y_20 border_y_24 border_y_32 }
    };
}

/// GPUI's border width scale (0 to 12, 16, 20, 24 and 32 px) for one group of fields.
macro_rules! border_scale {
    ($fields:tt $w0:ident $w1:ident $w2:ident $w3:ident $w4:ident $w5:ident $w6:ident $w7:ident $w8:ident
     $w9:ident $w10:ident $w11:ident $w12:ident $w16:ident $w20:ident $w24:ident $w32:ident) => {
        fixed_setters! { $fields {
            $w0 = px(0.), $w1 = px(1.), $w2 = px(2.), $w3 = px(3.), $w4 = px(4.), $w5 = px(5.), $w6 = px(6.),
            $w7 = px(7.), $w8 = px(8.), $w9 = px(9.), $w10 = px(10.), $w11 = px(11.), $w12 = px(12.),
            $w16 = px(16.), $w20 = px(20.), $w24 = px(24.), $w32 = px(32.),
        } }
    };
}

pub(super) use {
    border_methods, border_scale, fixed_setters, margin_methods, padding_methods, position_methods,
    rounded_methods, size_methods, value_setters,
};
