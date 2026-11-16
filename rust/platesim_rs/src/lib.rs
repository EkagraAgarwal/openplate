//! Measurement-only float64, C-contiguous kernels. No fast math or FMA contraction.
use numpy::{IntoPyArray, PyArray3, PyArrayMethods, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;

fn invalid(s: &str) -> PyErr {
    PyValueError::new_err(s.to_owned())
}
fn shape(a: &PyReadonlyArray3<'_, f64>, expected: &[usize]) -> PyResult<()> {
    if a.shape() != expected || !a.is_c_contiguous() {
        return Err(invalid("expected matching C-contiguous float64 arrays"));
    }
    Ok(())
}

// Used only in a colour pass on a bipartite stencil. Each task writes one distinct
// cell, reads its own value and opposite-colour neighbours only. A completed
// par_iter is the barrier between colours. No &[f64] aliases this pointer while
// it writes. rust-numpy rejects overlapping mutable borrows. The GIL is held;
// callers must not mutate buffers through concurrently running native code.
#[derive(Clone, Copy)]
struct Cells(*mut f64);
unsafe impl Send for Cells {}
unsafe impl Sync for Cells {}
impl Cells {
    fn get(self, q: usize) -> f64 {
        unsafe { *self.0.add(q) }
    }
    fn put(self, q: usize, v: f64) {
        unsafe {
            *self.0.add(q) = v;
        }
    }
}

fn each(n: usize, parallel: bool, f: impl Fn(usize) + Sync + Send) {
    if parallel {
        (0..n).into_par_iter().for_each(f);
    } else {
        (0..n).for_each(f);
    }
}

#[pyfunction]
#[pyo3(signature=(x, rhs, fx, fy, fz, base, diag, work, periodic, sweeps=2, rbgs=true, reverse=false, parallel=false))]
#[allow(clippy::too_many_arguments)]
fn smooth(
    x: Bound<'_, PyArray3<f64>>,
    rhs: Bound<'_, PyArray3<f64>>,
    fx: Bound<'_, PyArray3<f64>>,
    fy: Bound<'_, PyArray3<f64>>,
    fz: Bound<'_, PyArray3<f64>>,
    base: Bound<'_, PyArray3<f64>>,
    diag: Bound<'_, PyArray3<f64>>,
    work: Bound<'_, PyArray3<f64>>,
    periodic: [bool; 3],
    sweeps: usize,
    rbgs: bool,
    reverse: bool,
    parallel: bool,
) -> PyResult<()> {
    // rust-numpy's automatic argument conversion panics on overlapping mutable
    // arrays. Fallible borrows turn that into a regular Python ValueError.
    let borrow_error = |e: numpy::BorrowError| invalid(&e.to_string());
    let mut x = x.try_readwrite().map_err(borrow_error)?;
    let rhs = rhs.try_readonly().map_err(borrow_error)?;
    let fx = fx.try_readonly().map_err(borrow_error)?;
    let fy = fy.try_readonly().map_err(borrow_error)?;
    let fz = fz.try_readonly().map_err(borrow_error)?;
    let base = base.try_readonly().map_err(borrow_error)?;
    let diag = diag.try_readonly().map_err(borrow_error)?;
    let mut work = work.try_readwrite().map_err(borrow_error)?;
    let dims = x.shape().to_vec();
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
    if dims.contains(&0) || !x.is_c_contiguous() || !work.is_c_contiguous() || work.shape() != dims
    {
        return Err(invalid(
            "nonempty C-contiguous x and same-shaped work required",
        ));
    }
    for a in [&rhs, &base, &diag] {
        shape(a, &dims)?;
    }
    shape(&fx, &[nx + 1, ny, nz])?;
    shape(&fy, &[nx, ny + 1, nz])?;
    shape(&fz, &[nx, ny, nz + 1])?;
    let (b, fx, fy, fz, base, diag) = (
        rhs.as_slice()?,
        fx.as_slice()?,
        fy.as_slice()?,
        fz.as_slice()?,
        base.as_slice()?,
        diag.as_slice()?,
    );
    let xs = x.as_slice_mut()?;
    let ws = work.as_slice_mut()?;
    let u = Cells(xs.as_mut_ptr());
    let w = Cells(ws.as_mut_ptr());
    let odd = dims.iter().zip(periodic).any(|(n, p)| p && n % 2 == 1);
    let rbgs = rbgs && !odd;
    let sweeps = sweeps
        .checked_mul(if odd { 2 } else { 1 })
        .ok_or_else(|| invalid("sweeps overflow"))?;
    for _ in 0..sweeps {
        for pass in 0..if rbgs { 2 } else { 1 } {
            let color = if reverse { 1 - pass } else { pass };
            let out = if rbgs { u } else { w };
            let omega = if rbgs { 1.0 } else { 2.0 / 3.0 };
            each(nx * ny, parallel, |ij| {
                let (i, j) = (ij / ny, ij % ny);
                for k in 0..nz {
                    if rbgs && (i + j + k) % 2 != color {
                        continue;
                    }
                    let q = (i * ny + j) * nz + k;
                    let v = u.get(q);
                    let mut a = base[q] * v;
                    if i > 0 {
                        a += fx[q] * (v - u.get(q - ny * nz));
                    } else if periodic[0] && nx > 1 {
                        a += fx[j * nz + k] * (v - u.get(((nx - 1) * ny + j) * nz + k));
                    }
                    if i + 1 < nx {
                        a += fx[q + ny * nz] * (v - u.get(q + ny * nz));
                    } else if periodic[0] && nx > 1 {
                        a += fx[j * nz + k] * (v - u.get(j * nz + k));
                    }
                    let yq = (i * (ny + 1) + j) * nz + k;
                    if j > 0 {
                        a += fy[yq] * (v - u.get(q - nz));
                    } else if periodic[1] && ny > 1 {
                        a += fy[i * (ny + 1) * nz + k] * (v - u.get((i * ny + ny - 1) * nz + k));
                    }
                    if j + 1 < ny {
                        a += fy[yq + nz] * (v - u.get(q + nz));
                    } else if periodic[1] && ny > 1 {
                        a += fy[i * (ny + 1) * nz + k] * (v - u.get(i * ny * nz + k));
                    }
                    let zq = (i * ny + j) * (nz + 1) + k;
                    if k > 0 {
                        a += fz[zq] * (v - u.get(q - 1));
                    } else if periodic[2] && nz > 1 {
                        a += fz[(i * ny + j) * (nz + 1)] * (v - u.get(q + nz - 1));
                    }
                    if k + 1 < nz {
                        a += fz[zq + 1] * (v - u.get(q + 1));
                    } else if periodic[2] && nz > 1 {
                        a += fz[(i * ny + j) * (nz + 1)] * (v - u.get(q + 1 - nz));
                    }
                    out.put(q, v + omega * (b[q] - a) / diag[q]);
                }
            });
            if !rbgs {
                xs.copy_from_slice(ws);
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn sq(x: f64) -> f64 {
    x * x
}
#[inline(always)]
fn left(v: [f64; 5]) -> f64 {
    let [v1, v2, v3, v4, v5] = v;
    let s1 = 13.0 / 12.0 * sq(v1 - 2.0 * v2 + v3) + 0.25 * sq(v1 - 4.0 * v2 + 3.0 * v3);
    let s2 = 13.0 / 12.0 * sq(v2 - 2.0 * v3 + v4) + 0.25 * sq(v2 - v4);
    let s3 = 13.0 / 12.0 * sq(v3 - 2.0 * v4 + v5) + 0.25 * sq(3.0 * v3 - 4.0 * v4 + v5);
    let (a1, a2, a3) = (
        0.1 / sq(1e-6 + s1),
        0.6 / sq(1e-6 + s2),
        0.3 / sq(1e-6 + s3),
    );
    let p1 = v1 / 3.0 - 7.0 / 6.0 * v2 + 11.0 / 6.0 * v3;
    let p2 = -v2 / 6.0 + 5.0 / 6.0 * v3 + v4 / 3.0;
    let p3 = v3 / 3.0 + 5.0 / 6.0 * v4 - v5 / 6.0;
    (a1 * p1 + a2 * p2 + a3 * p3) / (a1 + a2 + a3)
}

#[pyfunction]
#[pyo3(signature=(psi, speed, dx, periodic, parallel=false))]
fn rhs<'py>(
    py: Python<'py>,
    psi: PyReadonlyArray3<'py, f64>,
    speed: PyReadonlyArray3<'py, f64>,
    dx: f64,
    periodic: [bool; 3],
    parallel: bool,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let dims = psi.shape().to_vec();
    if dims.contains(&0) || !psi.is_c_contiguous() || !dx.is_finite() || dx <= 0.0 {
        return Err(invalid("nonempty C array and positive finite dx required"));
    }
    shape(&speed, &dims)?;
    let (u, v) = (psi.as_slice()?, speed.as_slice()?);
    let (ny, nz) = (dims[1], dims[2]);
    let cell = |q: usize| {
        let loc = [q / (ny * nz), (q / nz) % ny, q % nz];
        let (mut gp, mut gn) = (0.0, 0.0);
        for axis in 0..3 {
            let mut nodes = [0.0; 7];
            for (r, node) in nodes.iter_mut().enumerate() {
                let mut ix = loc;
                let t = loc[axis] as isize + r as isize - 3;
                ix[axis] = if periodic[axis] {
                    t.rem_euclid(dims[axis] as isize) as usize
                } else {
                    t.clamp(0, dims[axis] as isize - 1) as usize
                };
                *node = u[(ix[0] * ny + ix[1]) * nz + ix[2]];
            }
            let d: [f64; 6] = std::array::from_fn(|r| (nodes[r + 1] - nodes[r]) / dx);
            let dm = left([d[0], d[1], d[2], d[3], d[4]]);
            let dp = -left([-d[5], -d[4], -d[3], -d[2], -d[1]]);
            gp += sq(dm.max(0.0)) + sq(dp.min(0.0));
            gn += sq(dm.min(0.0)) + sq(dp.max(0.0));
        }
        v[q].max(0.0) * gp.sqrt() + v[q].min(0.0) * gn.sqrt()
    };
    let out: Vec<f64> = if parallel {
        (0..u.len()).into_par_iter().map(cell).collect()
    } else {
        (0..u.len()).map(cell).collect()
    };
    Ok(
        numpy::ndarray::Array3::from_shape_vec((dims[0], ny, nz), out)
            .unwrap()
            .into_pyarray(py),
    )
}

#[pymodule]
fn platesim_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smooth, m)?)?;
    m.add_function(wrap_pyfunction!(rhs, m)?)?;
    m.add_function(wrap_pyfunction!(threads, m)?)?;
    Ok(())
}

#[pyfunction]
fn threads() -> usize {
    rayon::current_num_threads()
}
