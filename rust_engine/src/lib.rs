// rust_engine/src/main.rs
use ndarray::{Array2, ArrayView1, ArrayView2, s};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
// 添加到 lib.rs 顶部
use ndarray::ArrayViewMut2;

const EPS: f64 = 1e-12;

// === Rust 版 original_interaction ===
#[inline(always)]
fn original_interaction(
    xi: ArrayView1<f64>,
    xj: ArrayView1<f64>,
    si: ArrayView1<f64>,
    sj: ArrayView1<f64>,
    params: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let [A, B, J, K] = [params[0], params[1], params[2], params[3]];
    let rij: Vec<f64> = xj.iter().zip(xi.iter()).map(|(a, b)| a - b).collect();
    let dist_sq: f64 = rij.iter().map(|&r| r * r).sum();

    if dist_sq < EPS * EPS {
        return (vec![0.0; xi.len()], vec![0.0; si.len()]);
    }

    let dist = dist_sq.sqrt();
    let inv_dist = 1.0 / dist;
    let n_ij: Vec<f64> = rij.into_iter().map(|r| r * inv_dist).collect();

    let dot_s: f64 = si.iter().zip(sj.iter()).map(|(a, b)| a * b).sum();
    let mag_att = A + J * dot_s;
    let mag_rep = B * inv_dist;
    let f_pos: Vec<f64> = n_ij.iter().map(|&n| (mag_att - mag_rep) * n).collect();

    let proj_sj: Vec<f64> = sj
        .iter()
        .zip(si.iter())
        .map(|(sj_i, si_i)| sj_i - dot_s * si_i)
        .collect();
    let f_spin: Vec<f64> = proj_sj.into_iter().map(|p| K * inv_dist * p).collect();

    (f_pos, f_spin)
}

// === 计算所有粒子受力 ===
fn compute_all_forces(
    y: &ArrayView2<f64>,
    d: usize,
    d_s: usize,
    v0: &ArrayView2<f64>,
    omega0: &ArrayView2<f64>,
    params: &[f64],
) -> (Array2<f64>, Array2<f64>) {
    let n = y.nrows();
    let mut total_fx = Array2::<f64>::zeros((n, d));
    let mut total_fs = Array2::<f64>::zeros((n, d_s));

    (0..n).into_par_iter().for_each(|i| {
        let mut fx_i = vec![0.0; d];
        let mut fs_i = vec![0.0; d_s];

        // 零拷贝获取视图（无 .to_vec()）
        let xi = y.slice(s![i, 0..d]);
        let si = y.slice(s![i, 2 * d..2 * d + d_s]);

        for j in 0..n {
            if i == j {
                continue;
            }

            let xj = y.slice(s![j, 0..d]);
            let sj = y.slice(s![j, 2 * d..2 * d + d_s]);

            let (dv, ds) = original_interaction(xi, xj, si, sj, params);

            // 累加 dv → fx_i, ds → fs_i
            for (dst, &val) in fx_i.iter_mut().zip(dv.iter()) {
                *dst += val;
            }
            for (dst, &val) in fs_i.iter_mut().zip(ds.iter()) {
                *dst += val;
            }
        }

        let inv_n = 1.0 / (n as f64);
        for k in 0..d {
            total_fx[[i, k]] = fx_i[k] * inv_n + v0[[i, k]];
        }
        for k in 0..d_s {
            total_fs[[i, k]] = fs_i[k] * inv_n + omega0[[i, k]];
        }
    });

    (total_fx, total_fs)
}
// === 自旋归一化（原地）===

// 新增函数（放在 normalize_spin_inplace_impl 位置）
fn normalize_spin_inplace_impl(y: &mut ArrayViewMut2<f64>, d: usize, d_s: usize) {
    let start = 2 * d;
    let end = start + d_s;
    let mut spins = y.slice_mut(s![.., start..end]);
    spins
        .axis_iter_mut(ndarray::Axis(0))
        .par_bridge()
        .for_each(|mut s_vec| {
            let norm_sq = s_vec.iter().map(|&x| x * x).sum::<f64>();
            if norm_sq > 1e-24 {
                s_vec /= norm_sq.sqrt();
            }
        });
}

#[pyfunction]
#[pyo3(signature = (y, d, d_s))]
fn normalize_spins_inplace(
    mut y: PyRefMut<'_, PyArray2<f64>>,
    d: usize,
    d_s: usize,
) -> PyResult<()> {
    let mut view = y.as_array_mut().view_mut();
    normalize_spin_inplace_impl(&mut view, d, d_s);
    Ok(())
}

// === Rust 版 f_ode_dynamic ===
#[pyfunction]
#[pyo3(signature = (t, y, d, d_s, v0, omega0, coeff_mat, params))]
fn f_ode_dynamic_rust(
    _py: Python,
    _t: f64,
    y: PyReadonlyArray2<f64>,
    d: usize,
    d_s: usize,
    v0: PyReadonlyArray2<f64>,
    omega0: PyReadonlyArray2<f64>,
    coeff_mat: PyReadonlyArray2<f64>,
    params: Vec<f64>,
) -> PyResult<PyObject> {
    let y = y.as_array();
    let v0 = v0.as_array();
    let omega0 = omega0.as_array();
    let coeff_mat = coeff_mat.as_array();

    let (fx, fs) = compute_all_forces(&y, d, d_s, &v0, &omega0, &params);

    let n = y.nrows();
    let mut dydt = Array2::<f64>::zeros(y.raw_dim());

    dydt.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let vi = y.row(i).slice(s![d..2 * d]);
            let wi = y.row(i).slice(s![2 * d + d_s..]);

            let m = coeff_mat[[i, 0]];
            let beta_m = coeff_mat[[i, 1]];
            let ms = coeff_mat[[i, 2]];
            let beta_s = coeff_mat[[i, 3]];

            // 位置部分
            if m == 0.0 {
                row.slice_mut(s![0..d]).assign(&fx.row(i));
                row.slice_mut(s![d..2 * d]).fill(0.0);
            } else {
                row.slice_mut(s![0..d]).assign(&vi);
                let acc = (&fx.row(i) - beta_m * &vi) / m;
                row.slice_mut(s![d..2 * d]).assign(&acc);
            }

            // 自旋部分
            if ms == 0.0 {
                let s_curr = y.row(i).slice(s![2 * d..2 * d + d_s]);
                let ds_raw = fs.row(i);
                let dot = ds_raw.dot(&s_curr);
                let proj = &ds_raw - dot * &s_curr;
                row.slice_mut(s![2 * d..2 * d + d_s]).assign(&proj);
                row.slice_mut(s![2 * d + d_s..]).fill(0.0);
            } else {
                row.slice_mut(s![2 * d..2 * d + d_s]).assign(&wi);
                let torque = (&fs.row(i) - beta_s * &wi) / ms;
                row.slice_mut(s![2 * d + d_s..]).assign(&torque);
            }
        });

    Ok(dydt.to_pyarray(_py).to_owned())
}

#[pymodule]
fn rust_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(f_ode_dynamic_rust, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_spins_inplace, m)?)?; // ← 新增
    Ok(())
}
