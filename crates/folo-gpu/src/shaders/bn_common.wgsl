// Shared by the BatchNorm kernels. Spec: ACT (0 identity, 1 SiLU, 2 ReLU).
fn bn_act(z: vec4<f32>) -> vec4<f32> {
    switch ACT {
        case 1u: { return z / (vec4<f32>(1.0) + exp(-z)); }
        case 2u: { return max(z, vec4<f32>(0.0)); }
        default: { return z; }
    }
}

// The derivative of the activation at z.
fn bn_act_grad(z: vec4<f32>) -> vec4<f32> {
    switch ACT {
        case 1u: {
            let s = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-z));
            return s * (vec4<f32>(1.0) + z * (vec4<f32>(1.0) - s));
        }
        case 2u: { return select(vec4<f32>(0.0), vec4<f32>(1.0), z > vec4<f32>(0.0)); }
        default: { return vec4<f32>(1.0); }
    }
}

// Walks the elements [start, end) of channel c of an NCHW tensor (hw values per plane), 256 threads in a
// stride: element e is plane e / hw, offset e % hw. Params 5 and 6 hold 256 / hw and 256 % hw, so a step
// needs no division.
struct BnWalk { e: u32, n: u32, i: u32 }

fn bn_walk_start(start: u32, lid: u32, hw: u32) -> BnWalk {
    let e = start + lid;
    let n = e / hw;
    return BnWalk(e, n, e - n * hw);
}

fn bn_walk_next(w: ptr<function, BnWalk>, hw: u32) {
    (*w).e += 256u;
    (*w).n += pu(5u);
    (*w).i += pu(6u);
    if ((*w).i >= hw) {
        (*w).i -= hw;
        (*w).n += 1u;
    }
}
