// src/vcf_polarize.rs
use std::collections::{HashMap, HashSet};

use noodles_vcf as vcf;
use vcf::variant::record::info::field::key;
use vcf::variant::record::samples::keys::key as format_key;
use vcf::variant::record_buf::info::field::Value as InfoValue;
use vcf::variant::record_buf::samples::sample::Value as SampleValue;

use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::neldermead::NelderMead;
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use smallvec::SmallVec;

pub fn parsimony_ancestral<T>(
    header: &vcf::Header,
    record: &T,
    outgroup_aa: &HashSet<String>,
    ac: i32,
    af: f32,
    an: i32,
    ref_allel: &String,
    alt_allel: &String,
) -> Result<Option<vcf::variant::RecordBuf>, Box<dyn std::error::Error>>
where
    T: vcf::variant::Record,
{
    // If alt_allel is the ancestral one, we need to swap/polarize
    if outgroup_aa.contains(alt_allel) {
        let ac_swapped = an - ac;
        let af_swapped = 1.0 - af;

        let mut record_buf = vcf::variant::RecordBuf::try_from_variant_record(header, record)?;

        // Swap alleles
        *record_buf.reference_bases_mut() = alt_allel.to_string().into();
        *record_buf.alternate_bases_mut() = vec![ref_allel.to_string()].into();

        // Update INFO
        let info_mut = record_buf.info_mut();
        info_mut.insert(
            key::ALLELE_COUNT.to_string(),
            Some(InfoValue::Integer(ac_swapped)),
        );
        info_mut.insert(
            key::ALLELE_FREQUENCIES.to_string(),
            Some(InfoValue::Float(af_swapped)),
        );

        // Update Genotypes
        let samples_buf = std::mem::take(record_buf.samples_mut());
        let (keys, mut cols): (vcf::variant::record_buf::samples::Keys, _) = samples_buf.into();

        let gt_idx_opt = keys.as_ref().iter().position(|k| k == format_key::GENOTYPE);

        if let Some(gt_idx) = gt_idx_opt {
            for sample_fields in &mut cols {
                if let Some(Some(SampleValue::Genotype(genotype))) = sample_fields.get_mut(gt_idx) {
                    for allele in genotype.as_mut().iter_mut() {
                        if let Some(idx) = allele.position_mut() {
                            match *idx {
                                0 => *idx = 1,
                                1 => *idx = 0,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        *record_buf.samples_mut() = vcf::variant::record_buf::Samples::new(keys, cols);

        return Ok(Some(record_buf));
    }

    // Return None if no swap was performed (ref is ancestral or unknown)
    Ok(None)
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, clap::ValueEnum)]
pub enum PolarizationMethod {
    Parsimony,
    JC,
    Kimura,
    R6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Base {
    A = 0,
    C = 1,
    G = 2,
    T = 3,
    N = 4,
}

impl Base {
    pub fn from_char(c: char) -> Base {
        match c.to_ascii_uppercase() {
            'A' => Base::A,
            'C' => Base::C,
            'G' => Base::G,
            'T' => Base::T,
            '-' | '.' | 'N' => Base::N,
            _ => Base::N,
        }
    }

    pub fn from_str(s: &str) -> Base {
        s.chars().next().map(Base::from_char).unwrap_or(Base::N)
    }
}

pub fn base_to_index(b: Base) -> Option<usize> {
    match b {
        Base::A => Some(0),
        Base::C => Some(1),
        Base::G => Some(2),
        Base::T => Some(3),
        Base::N => None,
    }
}

pub fn base_to_states(bases: &[Base]) -> SmallVec<[i8; 8]> {
    bases
        .iter()
        .map(|&b| {
            // Use match to explicitly handle the conversion to i8
            match b {
                // A, C, G, T are 0, 1, 2, 3 (already positive, safe to cast to i8)
                Base::A => Base::A as i8,
                Base::C => Base::C as i8,
                Base::G => Base::G as i8,
                Base::T => Base::T as i8,
                _ => -1_i8,
            }
        })
        .collect()
}

pub fn base_to_counts(b: Base) -> [u8; 4] {
    let mut c = [0_u8; 4];
    if let Some(idx) = base_to_index(b) {
        c[idx] = 1;
    }
    c
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConfigKey {
    pub major: Base,          // major allele in ingroup sample
    pub minor: Base,          // minor allele in ingroup sample
    pub n_major: u16,         // copies of major allele
    pub n_total: u16,         // AN (total chromosomes)
    pub outgroups: Vec<Base>, // base per outgroup, in a fixed order
}

type Count = u32;

#[derive(Default, Debug)]
pub struct CollectedData {
    pub configs: HashMap<ConfigKey, Count>,
    pub n_sites_total: u64,
}

impl CollectedData {
    pub fn add_site(&mut self, key: ConfigKey) {
        *self.configs.entry(key).or_insert(0) += 1;
        self.n_sites_total += 1;
    }
}

// Vector of configs cheaper to iter than hashmap
#[derive(Default, Debug, Clone)]
pub struct ConfigRow {
    pub major: i16, // 0..3
    pub minor: i16, // 0..3,-1=missing
    pub n_major: u16,
    pub n_total: u16,
    pub outgroups: SmallVec<[i8; 8]>, // each 0..3, -1=missing
    pub multiplicity: u32,
}

// Phase 1: fitting subsitution model
#[derive(Debug, Clone)]
pub struct SubstitutionModelFit {
    pub method: PolarizationMethod,
    pub n_outgroups: usize,
    pub branch_k: Vec<f64>,         // length = 2*n_outgroups-1
    pub kappa: Option<f64>,         // Some(k) for Kimura, None for JC
    pub r6_rates: Option<[f64; 6]>, // NEW: Some(rates) for R6, None otherwise
    pub nll: f64,                   // best negative log-likelihood
    #[allow(dead_code)]
    pub z_hat: Vec<f64>, // best unconstrained params
}

#[derive(Clone)]
struct SubstitutionModelObjective {
    configs: Arc<Vec<ConfigRow>>,
    n_outgroups: usize,
    method: PolarizationMethod,

    // bounds
    kappa_min: f64,
    kappa_max: f64,
    k_min: f64,
    k_max: f64,
}

impl SubstitutionModelObjective {
    #[inline]
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    fn dim(&self) -> usize {
        let n_branches = 2 * self.n_outgroups - 1;
        match self.method {
            PolarizationMethod::JC => n_branches,
            PolarizationMethod::Kimura => n_branches + 1,
            PolarizationMethod::R6 => n_branches + 5,
            PolarizationMethod::Parsimony => 0,
        }
    }

    fn unpack_params(&self, z: &[f64]) -> (Vec<f64>, Option<f64>, Option<[f64; 6]>) {
        let n_branches = 2 * self.n_outgroups - 1;

        let mut ks = Vec::with_capacity(n_branches);
        for i in 0..n_branches {
            let s = Self::sigmoid(z[i]);
            ks.push(self.k_min + (self.k_max - self.k_min) * s);
        }

        let mut kappa = None;
        let mut r6_rates = None;

        match self.method {
            PolarizationMethod::Kimura => {
                let s = Self::sigmoid(z[n_branches]);
                kappa = Some(self.kappa_min + (self.kappa_max - self.kappa_min) * s);
            }
            PolarizationMethod::R6 => {
                let mut r = [0.0f64; 6];
                let mut sum = 0.0;

                for i in 0..5 {
                    r[i] = Self::sigmoid(z[n_branches + i]); // in (0,1)
                    sum += r[i];
                }

                // C behavior: infeasible => invalid (do NOT rescale)
                if sum >= 1.0 {
                    r6_rates = None;
                } else {
                    r[5] = 1.0 - sum;
                    r6_rates = Some(r);
                }
            }

            _ => {}
        }

        (ks, kappa, r6_rates)
    }

    fn build_cache(&self, ks: &[f64], kappa: Option<f64>, r6: Option<[f64; 6]>) -> Vec<f64> {
        let n_branches = ks.len();
        let mut cache = vec![0.0f64; n_branches * 16];

        for br in 0..n_branches {
            match self.method {
                PolarizationMethod::R6 => {
                    let rates = r6.expect("R6 rates missing in build_cache");
                    let pre = r6_precompute(ks[br], &rates);

                    for b1 in 0..4 {
                        for b2 in 0..4 {
                            let p = r6_prob_from_precomp(b1 as i8, b2 as i8, &pre);
                            cache[br * 16 + b1 * 4 + b2] = p.max(0.0);
                        }
                    }
                }
                PolarizationMethod::JC => {
                    for b1 in 0..4 {
                        for b2 in 0..4 {
                            cache[br * 16 + b1 * 4 + b2] =
                                jc_prob(b1 as i8, b2 as i8, ks[br]).max(0.0);
                        }
                    }
                }
                PolarizationMethod::Kimura => {
                    let kap = kappa.expect("kappa missing for Kimura");
                    for b1 in 0..4 {
                        for b2 in 0..4 {
                            cache[br * 16 + b1 * 4 + b2] =
                                k2_prob(b1 as i8, b2 as i8, ks[br], kap).max(0.0);
                        }
                    }
                }
                PolarizationMethod::Parsimony => unreachable!(),
            }
        }

        cache
    }

    /*#[inline]
    fn logsumexp2(a: f64, b: f64) -> f64 {
        if a == f64::NEG_INFINITY { return b; }
        if b == f64::NEG_INFINITY { return a; }
        let m = a.max(b);
        m + ((a - m).exp() + (b - m).exp()).ln()
    }

    fn log_p_tree(
        &self,
        focal: i8,
        internal: &[i8],
        out: &[i8],
        cache: &[f64],
    ) -> f64 {
        let n = self.n_outgroups;
        let n_branches = 2 * n - 1;
        let mut lp = 0.0f64;

        for i in 0..n_branches {
            let (b1, b2) = if i == 0 {
                let rhs = if n == 1 { out[0] } else { internal[0] };
                (focal, rhs)
            } else if i < n_branches - 1 {
                let i_internal = (i - 1) / 2;
                let b1 = internal[i_internal];
                let b2 = if i % 2 == 1 { out[i_internal] } else { internal[i_internal + 1] };
                (b1, b2)
            } else {
                (internal[internal.len() - 1], out[out.len() - 1])
            };

            if b1 < 0 || b2 < 0 {
                //return f64::NEG_INFINITY;
                continue
            }

            let idx = i * 16 + (b1 as usize) * 4 + (b2 as usize);
            let p = cache[idx];
            if p <= 0.0 {
                return f64::NEG_INFINITY;
            }
            lp += p.ln();
        }

        lp
    }*/
}

fn p_config(n_outgroups: usize, focal: i16, out: &[i8], cache: &[f64]) -> f64 {
    if focal < 0 {
        return 0.0;
    }
    let focal = focal as i8;

    match n_outgroups {
        1 => {
            // branch 0: internal==out[0] directly in n=1 case
            p_branch(cache, 0, focal, out[0])
        }
        2 => {
            // C: internal node b2; branches: (b2->focal)=0, (b2->out1)=1, (b2->out2)=2
            let o1 = out[0];
            let o2 = out[1];
            let mut tot = 0.0;
            for b2 in 0..4 {
                let b2 = b2 as i8;
                let p = p_branch(cache, 0, focal, b2)
                    * p_branch(cache, 1, b2, o1)
                    * p_branch(cache, 2, b2, o2);
                tot += p;
            }
            tot
        }
        3 => {
            // C: internal nodes b2 (near ingroup) and b4 (deeper)
            // branches: 0:(b2->focal), 1:(b2->out1), 2:(b4->b2), 3:(b4->out2), 4:(b4->out3)
            let o1 = out[0];
            let o2 = out[1];
            let o3 = out[2];
            let mut tot = 0.0;

            for b2u in 0..4 {
                let b2 = b2u as i8;
                for b4u in 0..4 {
                    let b4 = b4u as i8;
                    let p = p_branch(cache, 0, focal, b2) *
                        p_branch(cache, 1, b2, o1) *
                        p_branch(cache, 2, b2, b4) *  // orientation irrelevant under your symmetric probs
                        p_branch(cache, 3, b4, o2) *
                        p_branch(cache, 4, b4, o3);
                    tot += p;
                }
            }
            tot
        }
        _ => {
            // keep your existing generic implementation for >3 later,
            // but for now you said 1..3 is your focus.
            0.0
        }
    }
}

impl CostFunction for SubstitutionModelObjective {
    type Param = Vec<f64>; // unconstrained z
    type Output = f64; // NLL

    fn cost(&self, z: &Self::Param) -> Result<Self::Output, Error> {
        let (ks, kappa, r6_rates) = self.unpack_params(z);

        // C behavior: invalid params => -inf ln_l => +inf NLL
        if self.method == PolarizationMethod::R6 {
            let r6 = match r6_rates {
                Some(r) => r,
                None => return Ok(1e300), // infeasible => -inf ln_l => +inf NLL
            };

            // C also effectively rejects boundary solutions
            if !r6.iter().all(|x| x.is_finite() && *x > 0.0) {
                return Ok(1e300);
            }
        }

        let cache = self.build_cache(&ks, kappa, r6_rates);
        //let cache = self.build_cache(&ks, kappa, r6_rates);

        let mut nll = 0.0f64;

        for v in self.configs.iter() {
            if v.outgroups.len() != self.n_outgroups {
                continue;
            }

            let p_major = p_config(self.n_outgroups, v.major, &v.outgroups, &cache);

            let p_site = if v.n_major == v.n_total {
                // fixed/monomorphic category: C ignores the minor likelihood
                p_major
            } else {
                let p_minor = p_config(self.n_outgroups, v.minor, &v.outgroups, &cache);
                0.5 * (p_major + p_minor)
            };

            // guard
            let p_site = p_site.max(1e-300);
            nll -= (v.multiplicity as f64) * p_site.ln();
        }

        Ok(nll)
    }

    /*  fn cost(&self, z: &Self::Param) -> Result<Self::Output, Error> {
        let (ks, k) = self.unpack_params(z);
        let cache = self.build_cache(&ks, k);

        let ln2 = 2.0f64.ln();
        let mut nll = 0.0f64;

        for v in self.configs.iter() {
            if v.outgroups.len() != self.n_outgroups { continue; }

            let lp_major = self.log_p_config(v.major, &v.outgroups, &cache);
            let lp_minor = self.log_p_config(v.minor, &v.outgroups, &cache);
            /*let lp_site = SubstitutionModelObjective::logsumexp2(lp_major, lp_minor) - ln2;*/
            let lp_site = if v.minor < 0 {
                lp_major
            } else {
                SubstitutionModelObjective::logsumexp2(lp_major, lp_minor) - ln2
            };
            if !lp_site.is_finite() {
                return Ok(1e300);
            }
            nll -= (v.multiplicity as f64) * lp_site;
        }

        Ok(nll)
    }*/
}

#[inline]
fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

#[inline(always)]
fn p_branch(cache: &[f64], branch: usize, a: i8, b: i8) -> f64 {
    if a < 0 || b < 0 {
        1.0 // C behavior: missing endpoint contributes 1.0
    } else {
        cache[branch * 16 + (a as usize) * 4 + (b as usize)]
    }
}

#[inline]
fn jc_prob(b1: i8, b2: i8, k: f64) -> f64 {
    let e = (-k).exp();
    if b1 == b2 {
        e + (1.0 / 6.0) * k * k * e
    } else {
        (1.0 / 3.0) * k * e + (1.0 / 9.0) * k * k * e
    }
}

#[inline]
fn is_transition(b1: i8, b2: i8) -> bool {
    matches!((b1, b2), (0, 2) | (2, 0) | (1, 3) | (3, 1))
}

#[inline]
fn k2_prob(b1: i8, b2: i8, k: f64, kappa: f64) -> f64 {
    let e = (-k).exp();
    let denom = (kappa + 2.0) * (kappa + 2.0);
    if b1 == b2 {
        e * (1.0 + 0.5 * k * k * (2.0 + kappa * kappa) / denom)
    } else if is_transition(b1, b2) {
        k * e * (kappa / (kappa + 2.0) + k * (1.0 / denom))
    } else {
        k * e * (1.0 / (kappa + 2.0) + k * (kappa / denom))
    }
}

fn initial_simplex(x0: &[f64], step: f64) -> Vec<Vec<f64>> {
    let mut simplex = Vec::with_capacity(x0.len() + 1);
    simplex.push(x0.to_vec());
    for i in 0..x0.len() {
        let mut xi = x0.to_vec();
        xi[i] += step;
        simplex.push(xi);
    }
    simplex
}

#[inline]
fn z_from_bounded_value(x: f64, min: f64, max: f64) -> f64 {
    // Inverse of: x = min + (max-min) * sigmoid(z)
    // so: sigmoid(z) = (x-min)/(max-min)
    // and: z = logit(sigmoid(z))
    let eps = 1e-12;

    let mut s = (x - min) / (max - min);

    // Keep s strictly inside (0,1) so logit() stays finite
    if s <= eps {
        s = eps;
    }
    if s >= 1.0 - eps {
        s = 1.0 - eps;
    }

    logit(s)
}

pub fn fit_substitution_model(
    variant_configs: Vec<ConfigRow>,
    n_outgroups: usize,
    method: PolarizationMethod,
    nrandom: usize,
    seed: u64,
) -> Result<SubstitutionModelFit, String> {
    if method == PolarizationMethod::Parsimony {
        return Err("Stage 1 rate inference is not applicable to Parsimony.".to_string());
    }
    if n_outgroups == 0 || n_outgroups > 8 {
        return Err("n_outgroups must be in 1..=8".to_string());
    }

    let filtered: Vec<ConfigRow> = variant_configs
        .into_iter()
        .filter(|v| v.outgroups.len() == n_outgroups)
        // .filter(|v| v.outgroups.iter().all(|&b| b >= 0)) // Python MLE set
        .collect();

    if filtered.is_empty() {
        return Err("No configs after filtering (n_outgroups / missing outgroups).".to_string());
    }

    let op = SubstitutionModelObjective {
        configs: Arc::new(filtered),
        n_outgroups,
        method,
        k_min: 1e-4,
        k_max: 10.0,
        kappa_min: 1e-3,
        kappa_max: 10.0,
    };

    let dim = op.dim();
    let mut rng = StdRng::seed_from_u64(seed);

    let mut best_z = vec![0.0f64; dim];
    let mut best_nll = f64::INFINITY;
    let n_branches = 2 * n_outgroups - 1;

    for _i in 0..nrandom {
        let mut z0 = vec![0.0f64; dim];

        /*for zi in &mut z0 {
            *zi = rng.random_range(-4.0..4.0);
        }*/

        for j in 0..n_branches {
            // [0, 0.2)
            let k0 = rng.random::<f64>() * 0.2;
            // Your bounds are [op.k_min, op.k_max]
            let k0 = k0.clamp(op.k_min, op.k_max);
            z0[j] = z_from_bounded_value(k0, op.k_min, op.k_max);
        }

        // C: if kimura: kappa = uniform() * 10
        if method == PolarizationMethod::Kimura {
            let kap0 = rng.random::<f64>() * 10.0; // [0, 10)
            let kap0 = kap0.clamp(op.kappa_min, op.kappa_max);
            z0[n_branches] = z_from_bounded_value(kap0, op.kappa_min, op.kappa_max);
        }

        if method == PolarizationMethod::R6 {
            // pick small values so sum < 1 safely
            // e.g. 5 * 0.05 = 0.25, so r6[5] = 0.75
            let base = 0.05;
            for i in 0..5 {
                z0[n_branches + i] = logit(base);
            }
        }

        let simplex = initial_simplex(&z0, 0.5);
        let solver = NelderMead::new(simplex);

        let res = Executor::new(op.clone(), solver)
            .configure(|state| state.max_iters(600))
            .run()
            .map_err(|e| format!("argmin run failed: {e}"))?;

        let z_hat = res
            .state()
            .best_param
            .as_ref()
            .ok_or_else(|| "No best_param returned by argmin".to_string())?
            .clone();

        let nll = res.state().best_cost;

        if nll < best_nll {
            best_nll = nll;
            best_z = z_hat;
        }
    }

    let (ks, kappa, r6_rates) = op.unpack_params(&best_z);

    Ok(SubstitutionModelFit {
        method,
        n_outgroups,
        branch_k: ks,
        kappa,
        r6_rates,
        nll: best_nll,
        z_hat: best_z,
    })
}

// r6 model
#[inline]
fn r6_type(b1: i8, b2: i8) -> usize {
    match (b1, b2) {
        (0, 3) | (3, 0) => 0, // A<->T
        (0, 1) | (1, 0) => 1, // A<->C
        (0, 2) | (2, 0) => 2, // A<->G
        (3, 1) | (1, 3) => 3, // T<->C
        (3, 2) | (2, 3) => 4, // T<->G
        (1, 2) | (2, 1) => 5, // C<->G
        _ => unreachable!(),
    }
}

#[inline]
fn lookuprate(b1: i8, b2: i8, r6: &[f64; 6]) -> f64 {
    r6[r6_type(b1, b2)]
}

#[inline]
fn intfunc(k1: f64, k2: f64) -> f64 {
    let mut diff = k1 - k2;
    if diff < 0.0 {
        diff = -diff;
    }

    if diff < 1e-6 {
        (k1 * k1 * (-k1).exp()) / 2.0
    } else {
        let num = k1 * k2 * (-(k2 + k1)).exp() * (k2.exp() - k1.exp() * k2 + (k1 - 1.0) * k1.exp());
        let denom = k2 * k2 - 2.0 * k1 * k2 + k1 * k1;
        num / denom
    }
}

fn p_for_two_changes(b1: i8, b2: i8, r6: &[f64; 6], k: f64) -> f64 {
    let mut totk = [0.0f64; 4];

    for i in 0..4i8 {
        for j in 0..4i8 {
            if i == j {
                continue;
            }
            let y = lookuprate(i, j, r6) * k * 2.0;
            totk[i as usize] += y;
        }
    }

    let mut totp = 0.0;
    for mid in 0..4i8 {
        if mid == b1 || mid == b2 {
            continue;
        }

        let k1 = lookuprate(b1, mid, r6) * k * 2.0;
        if k1 == 0.0 {
            continue;
        }

        let k2 = lookuprate(mid, b2, r6) * k * 2.0;
        if k2 == 0.0 {
            continue;
        }

        let mut p = intfunc(totk[b1 as usize], totk[mid as usize]);
        p /= totk[b1 as usize] / k1;
        p /= totk[mid as usize] / k2;
        totp += p;
    }

    totp
}

fn compute_poiss_r6_p2_change(r6_ind: usize, r6: &[f64; 6], k: f64) -> f64 {
    let (b1, b2) = match r6_ind {
        0 => (0i8, 3i8),
        1 => (0i8, 1i8),
        2 => (0i8, 2i8),
        3 => (3i8, 1i8),
        4 => (3i8, 2i8),
        5 => (1i8, 2i8),
        _ => unreachable!(),
    };

    let p1 = p_for_two_changes(b1, b2, r6, k);
    let p2 = p_for_two_changes(b2, b1, r6, k);
    (p1 + p2) / 4.0
}

fn compute_poiss_r6_p2_no_change(k: f64, r6: &[f64; 6]) -> f64 {
    let mut totk = [0.0f64; 4];

    for i in 0..4i8 {
        for j in 0..4i8 {
            if i == j {
                continue;
            }
            let y = lookuprate(i, j, r6) * k * 2.0;
            totk[i as usize] += y;
        }
    }

    let mut res = 0.0;
    for i in 0..4i8 {
        let mut totp = 0.0;

        for j in 0..4i8 {
            if i == j {
                continue;
            }

            let k1 = lookuprate(i, j, r6) * k * 2.0;
            if k1 == 0.0 {
                continue;
            }

            let mut p = intfunc(totk[i as usize], totk[j as usize]);
            p /= totk[i as usize] / k1;
            p /= totk[j as usize] / k1;
            totp += p;
        }

        res += totp;
    }

    res / 4.0
}

#[inline]
fn r6_p1_contrib(ind: usize, r: &[[f64; 3]; 2], k: f64) -> f64 {
    let kr6 = r[ind][0] + r[ind][1] + r[ind][2];
    if r[ind][0] == 0.0 {
        return 0.0;
    }

    let p = r[ind][0] / kr6;
    let kr6 = kr6 * 2.0 * k;
    (-kr6).exp() * kr6 * p
}

fn compute_r6_p1(r6_ind: usize, r6: &[f64; 6], k: f64) -> f64 {
    let mut r = [[0.0f64; 3]; 2];
    r[0][0] = r6[r6_ind];
    r[1][0] = r6[r6_ind];

    match r6_ind {
        0 => {
            r[0][1] = r6[1];
            r[0][2] = r6[2];
            r[1][1] = r6[3];
            r[1][2] = r6[4];
        }
        1 => {
            r[0][1] = r6[0];
            r[0][2] = r6[2];
            r[1][1] = r6[3];
            r[1][2] = r6[5];
        }
        2 => {
            r[0][1] = r6[0];
            r[0][2] = r6[1];
            r[1][1] = r6[4];
            r[1][2] = r6[5];
        }
        3 => {
            r[0][1] = r6[0];
            r[0][2] = r6[4];
            r[1][1] = r6[1];
            r[1][2] = r6[5];
        }
        4 => {
            r[0][1] = r6[0];
            r[0][2] = r6[3];
            r[1][1] = r6[2];
            r[1][2] = r6[5];
        }
        5 => {
            r[0][1] = r6[1];
            r[0][2] = r6[3];
            r[1][1] = r6[2];
            r[1][2] = r6[4];
        }
        _ => unreachable!(),
    }

    let c0 = r6_p1_contrib(0, &r, k);
    let c1 = r6_p1_contrib(1, &r, k);
    (c0 + c1) / 4.0
}

#[derive(Copy, Clone)]
struct R6BranchPrecomp {
    poiss_mean: f64,
    p2_no_sub: f64,
    p1: [f64; 6],
    p2: [f64; 6],
}

fn r6_precompute(k: f64, r6: &[f64; 6]) -> R6BranchPrecomp {
    // base-wise total outgoing rate sums:
    // A: (AT, AC, AG) => r6[0]+r6[1]+r6[2]
    // T: (TA, TC, TG) => r6[0]+r6[3]+r6[4]
    // C: (CA, CT, CG) => r6[1]+r6[3]+r6[5]
    // G: (GA, GT, GC) => r6[2]+r6[4]+r6[5]
    let poiss_a = (-(k * 2.0) * (r6[0] + r6[1] + r6[2])).exp();
    let poiss_t = (-(k * 2.0) * (r6[0] + r6[3] + r6[4])).exp();
    let poiss_c = (-(k * 2.0) * (r6[1] + r6[3] + r6[5])).exp();
    let poiss_g = (-(k * 2.0) * (r6[2] + r6[4] + r6[5])).exp();
    let poiss_mean = (poiss_a + poiss_c + poiss_g + poiss_t) / 4.0;

    let p2_no_sub = compute_poiss_r6_p2_no_change(k, r6);

    let mut p1 = [0.0f64; 6];
    let mut p2 = [0.0f64; 6];
    for t in 0..6 {
        p1[t] = compute_r6_p1(t, r6, k);
        p2[t] = compute_poiss_r6_p2_change(t, r6, k);
    }

    R6BranchPrecomp {
        poiss_mean,
        p2_no_sub,
        p1,
        p2,
    }
}

#[inline]
fn r6_prob_from_precomp(b1: i8, b2: i8, pre: &R6BranchPrecomp) -> f64 {
    if b1 == b2 {
        pre.poiss_mean + pre.p2_no_sub
    } else {
        let t = r6_type(b1, b2);
        2.0 * (pre.p1[t] + pre.p2[t])
    }
}

// Phase 2: estimate uSFS from fitted model
#[derive(Debug, Clone)]
pub struct UsfsFit {
    pub n_total: u16,
    /// w[c] = P(major is ancestral | minor_count=c). Index 0..=floor(n/2).
    #[allow(dead_code)]
    pub weights_by_minor_count: Vec<f64>,
    /// Expected number of sites in each derived count (0..=n). Monomorphic classes (0,n) left as 0 here.
    pub expected_derived_counts: Vec<f64>,
    /// Normalized uSFS over derived counts (0..=n). Here 0,n will be 0 unless you add a monomorphic model.
    pub usfs: Vec<f64>,
    /// Log-likelihood at the fitted weights (polymorphic sites only).
    #[allow(dead_code)]
    pub ln_l: f64,
}

fn build_branch_transition_cache(fit: &SubstitutionModelFit) -> Vec<f64> {
    let ks = &fit.branch_k;
    let n_branches = ks.len();
    let mut cache = vec![0.0f64; n_branches * 16];

    match fit.method {
        PolarizationMethod::JC => {
            for br in 0..n_branches {
                for b1 in 0..4 {
                    for b2 in 0..4 {
                        cache[br * 16 + b1 * 4 + b2] = jc_prob(b1 as i8, b2 as i8, ks[br]).max(0.0);
                    }
                }
            }
        }
        PolarizationMethod::Kimura => {
            let kappa = fit
                .kappa
                .expect("SubstitutionModelFit.kappa missing for Kimura");
            for br in 0..n_branches {
                for b1 in 0..4 {
                    for b2 in 0..4 {
                        cache[br * 16 + b1 * 4 + b2] =
                            k2_prob(b1 as i8, b2 as i8, ks[br], kappa).max(0.0);
                    }
                }
            }
        }
        PolarizationMethod::R6 => {
            let r6 = fit
                .r6_rates
                .expect("SubstitutionModelFit.r6_rates missing for R6");
            for br in 0..n_branches {
                let k = ks[br];
                let pre = r6_precompute(k, &r6);
                for b1 in 0..4 {
                    for b2 in 0..4 {
                        let p = r6_prob_from_precomp(b1 as i8, b2 as i8, &pre);
                        cache[br * 16 + b1 * 4 + b2] = p.max(0.0);
                    }
                }
            }
        }
        _ => panic!("Stage 2 currently implemented for JC/Kimura/r6 only"),
    }

    cache
}

pub fn fit_usfs_minor_count(
    variant_configs: &[ConfigRow],
    n_outgroups: usize,
    stage1: &SubstitutionModelFit,
    max_iter: usize,
    tol: f64,
) -> UsfsFit {
    let cache = build_branch_transition_cache(stage1);

    // Infer n_total (assume constant for now; if not, you should split by n_total).
    let n_total = variant_configs
        .iter()
        .find(|v| v.n_total > 0)
        .map(|v| v.n_total)
        .unwrap_or(0);

    let n = n_total as usize;
    let max_mac = n / 2;

    // Per MAC bin: store (p_major, p_minor, mult, derived_minor_count=c)
    let mut bins: Vec<Vec<(f64, f64, u32, usize)>> = vec![Vec::new(); max_mac + 1];

    let mut fixed_sites: Vec<(f64, f64, u32)> = Vec::new();

    for v in variant_configs {
        if v.outgroups.len() != n_outgroups {
            continue;
        }
        if v.n_total == 0 {
            continue;
        }
        /*if v.minor < 0 {
            // monomorphic/fixed: Stage 2 mixture is undefined; handle later if you add f0/fn model
            continue;
        }*/
        if v.minor < 0 {
            // Fixed / monomorphic: derive posterior that the observed base is ancestral.
            let a = v.major as i8;
            debug_assert!((0..4).contains(&(a as i32)));

            let l_a = p_config(n_outgroups, a as i16, &v.outgroups, &cache).max(1e-300);

            let mut sum_other = 0.0;
            for b in 0..4i8 {
                if b == a {
                    continue;
                }
                let l_b = p_config(n_outgroups, b as i16, &v.outgroups, &cache).max(1e-300);
                sum_other += l_b;
            }
            let l_other_mean = (sum_other / 3.0).max(1e-300);

            fixed_sites.push((l_a, l_other_mean, v.multiplicity));
            continue;
        }
        let n_tot = v.n_total as usize;
        let n_maj = v.n_major as usize;
        let n_min = n_tot.saturating_sub(n_maj);

        // Define MAC as the minor allele count (as constructed in your pipeline).
        let c = n_min.min(n_tot - n_min);

        if c == 0 {
            continue;
        }
        if c > max_mac {
            continue;
        }

        let p_major = p_config(n_outgroups, v.major, &v.outgroups, &cache).max(1e-300);
        let p_minor = p_config(n_outgroups, v.minor, &v.outgroups, &cache).max(1e-300);

        bins[c].push((p_major, p_minor, v.multiplicity, c));
    }

    let mut weights_by_minor_count = vec![0.5f64; max_mac + 1];
    let mut expected = vec![0.0f64; n + 1];
    let mut ln_l = 0.0f64;

    for c in 1..=max_mac {
        if bins[c].is_empty() {
            continue;
        }

        // If n even and c == n/2: derived count is the same either way (c == n-c).
        // Weight is not identifiable; set 0.5 and just accumulate all mass to count=c.
        if 2 * c == n {
            weights_by_minor_count[c] = 0.5;
            for (p_m, p_n, mult, _) in &bins[c] {
                // any mixture gives same derived count, but likelihood still depends on mixture.
                // Using 0.5 is conventional here:
                let p_site = 0.5 * (*p_m + *p_n);
                ln_l += (*mult as f64) * p_site.ln();
                expected[c] += *mult as f64;
            }
            continue;
        }

        // EM for w in (0,1): w = mean posterior gamma
        let mut w = 0.5f64;

        for _ in 0..max_iter {
            let mut num = 0.0f64;
            let mut den = 0.0f64;

            for (p_m, p_n, mult, _) in &bins[c] {
                let m = *mult as f64;
                let mix = w * (*p_m) + (1.0 - w) * (*p_n);
                // posterior P(major ancestral)
                let gamma = (w * (*p_m)) / mix;
                num += m * gamma;
                den += m;
            }

            let w_new = (num / den).clamp(1e-12, 1.0 - 1e-12);
            if (w_new - w).abs() < tol {
                w = w_new;
                break;
            }
            w = w_new;
        }

        weights_by_minor_count[c] = w;

        // Accumulate expected uSFS and ln_l at fitted w
        for (p_m, p_n, mult, _) in &bins[c] {
            let m = *mult as f64;
            let mix = w * (*p_m) + (1.0 - w) * (*p_n);
            let gamma = (w * (*p_m)) / mix;

            // If major ancestral => derived count = c (minor count)
            expected[c] += m * gamma;
            // Else derived count = n - c
            expected[n - c] += m * (1.0 - gamma);

            ln_l += m * mix.ln();
        }
    }

    /*// Add fixed-site expected mass to bins 0 and n, and add their ln_l contribution.
    for (l_a, l_other_mean, mult) in fixed_sites.iter().copied() {
        let m = mult as f64;

        // Posterior P(major ancestral) under 0.5/0.5 mixture:
        let denom = l_a + l_other_mean;
        let post = l_a / denom;

        // Expected uSFS counts:
        expected[0] += m * post;           // ancestral fixed
        expected[n] += m * (1.0 - post);   // derived fixed

        // Likelihood for fixed sites under same mixture:
        let mix = 0.5 * l_a + 0.5 * l_other_mean;
        ln_l += m * mix.ln();
    }*/

    // Add fixed-site expected mass to bins 0 and n, and add their ln_l contribution.
    for (l_a, l_other_mean, mult) in fixed_sites.iter().copied() {
        let m = mult as f64;

        // C-like uniform root prior:
        // P(root=major)=1/4, P(root in other 3)=3/4 with mean likelihood l_other_mean
        let denom = 0.25 * l_a + 0.75 * l_other_mean;

        // Posterior P(major ancestral)
        let post = (0.25 * l_a) / denom;

        expected[0] += m * post;
        expected[n] += m * (1.0 - post);

        // ln_l contribution (optional; include if you want Stage2 ln_l to match C overall):
        ln_l += m * denom.ln();
    }

    // Normalize polymorphic spectrum into probabilities (0 and n remain 0 here)
    //let poly_sum: f64 = expected.iter().sum();
    let total_sum: f64 = expected.iter().sum();
    let mut usfs = vec![0.0f64; n + 1];
    if total_sum > 0.0 {
        for d in 0..=n {
            usfs[d] = expected[d] / total_sum;
            //usfs[d] = expected[d] / poly_sum;
        }
    }

    UsfsFit {
        n_total,
        weights_by_minor_count,
        expected_derived_counts: expected,
        usfs,
        ln_l,
    }
}


/*pub fn fit_usfs_minor_count(
    variant_configs: &[ConfigRow],
    n_outgroups: usize,
    stage1: &SubstitutionModelFit,
    max_iter: usize,
    tol: f64,
) -> UsfsFit {
    let cache = build_branch_transition_cache(stage1);

    // 1. Determine global N
    let n_total = variant_configs
        .iter()
        .find(|v| v.n_total > 0)
        .map(|v| v.n_total)
        .unwrap_or(0);

    let n = n_total as usize;
    let max_mac = n / 2;

    let mut bins: Vec<Vec<(f64, f64, u32, usize)>> = vec![Vec::new(); max_mac + 1];
    let mut fixed_sites: Vec<(f64, f64, u32)> = Vec::new();

    // --- DEBUG COUNTERS ---
    let mut count_fixed = 0;
    let mut count_poly = 0;
    let mut count_skipped_an = 0;

    for v in variant_configs {
        if v.outgroups.len() != n_outgroups || v.n_total == 0 {
            continue;
        }

        // Check if sample size matches the global N
        if v.n_total as usize != n {
            count_skipped_an += 1;
            continue;
        }

        if v.minor < 0 {
            // This is a fixed site
            let a = v.major as i8;
            let l_a = p_config(n_outgroups, a as i16, &v.outgroups, &cache).max(1e-300);
            
            let mut sum_other = 0.0;
            for b in 0..4i8 {
                if b == a { continue; }
                sum_other += p_config(n_outgroups, b as i16, &v.outgroups, &cache).max(1e-300);
            }
            let l_other_mean = (sum_other / 3.0).max(1e-300);

            fixed_sites.push((l_a, l_other_mean, v.multiplicity));
            count_fixed += 1;
            continue;
        }

        // This is a polymorphic site
        let n_tot = v.n_total as usize;
        let n_maj = v.n_major as usize;
        let n_min = n_tot.saturating_sub(n_maj);
        let c = n_min.min(n_tot - n_min);

        if c > 0 && c <= max_mac {
            let p_major = p_config(n_outgroups, v.major, &v.outgroups, &cache).max(1e-300);
            let p_minor = p_config(n_outgroups, v.minor, &v.outgroups, &cache).max(1e-300);
            bins[c].push((p_major, p_minor, v.multiplicity, c));
            count_poly += 1;
        }
    }

    let total_variants_in_bins: usize = bins.iter().map(|b| b.len()).sum();
    println!("Total SNPs in bins: {}", total_variants_in_bins);

    tracing::info!(
        "USFS Fitter Input: {} polymorphic, {} fixed, {} skipped (AN mismatch)",
        count_poly, count_fixed, count_skipped_an
    );

    let mut expected = vec![0.0f64; n + 1];
    let mut weights_by_minor_count = vec![0.5f64; max_mac + 1];
    let mut ln_l = 0.0f64;

    // 2. EM Loop
    for c in 1..=max_mac {
        if bins[c].is_empty() { continue; }

        if 2 * c == n {
            for (p_m, p_n, mult, _) in &bins[c] {
                let p_site = 0.5 * (*p_m + *p_n);
                ln_l += (*mult as f64) * p_site.ln();
                expected[c] += *mult as f64;
            }
            continue;
        }

        let mut w = 0.5f64;
        for _ in 0..max_iter {
            let (mut num, mut den) = (0.0, 0.0);
            for (p_m, p_n, mult, _) in &bins[c] {
                let m = *mult as f64;
                let mix = w * (*p_m) + (1.0 - w) * (*p_n);
                num += m * (w * (*p_m) / mix);
                den += m;
            }
            let w_new = (num / den).clamp(1e-12, 1.0 - 1e-12);
            if (w_new - w).abs() < tol { w = w_new; break; }
            w = w_new;
        }
        weights_by_minor_count[c] = w;

        for (p_m, p_n, mult, _) in &bins[c] {
            let m = *mult as f64;
            let mix = w * (*p_m) + (1.0 - w) * (*p_n);
            let gamma = (w * (*p_m)) / mix;
            expected[c] += m * gamma;
            expected[n - c] += m * (1.0 - gamma);
            ln_l += m * mix.ln();
        }
    }

    // Fixed sites logic
    for (l_a, l_other_mean, mult) in fixed_sites.iter().copied() {
        let m = mult as f64;
        let denom = 0.25 * l_a + 0.75 * l_other_mean;
        let post = (0.25 * l_a) / denom;
        expected[0] += m * post;
        expected[n] += m * (1.0 - post);
        ln_l += m * denom.ln();
    }

    // Normalization Fix: Calculate USFS based on POLYMORPHIC sites only
    let poly_sum: f64 = expected[1..n].iter().sum();
    let mut usfs = vec![0.0f64; n + 1];
    
    if poly_sum > 0.0 {
        for d in 1..n {
            usfs[d] = expected[d] / poly_sum;
        }
        tracing::info!("USFS Fit successful. Total polymorphic mass: {:.2}", poly_sum);
    } else {
        tracing::warn!("USFS Fit Warning: Zero polymorphic mass. Check VCF AC/AN fields.");
    }

    UsfsFit {
        n_total,
        weights_by_minor_count,
        expected_derived_counts: expected,
        usfs,
        ln_l,
    }
}*/

// Phase 3: Estimate ancestral posterior probabilites for each site
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct PosKey {
    pub chrom: String,
    pub pos: u32,
}

fn post_major_anc_polymorphic(
    n: usize,
    c: usize,
    l_major: f64,
    l_minor: f64,
    usfs: &[f64], // length n+1
) -> f64 {
    let pi_major = usfs[c].max(1e-300);
    let pi_minor = usfs[n - c].max(1e-300);
    let num = pi_major * l_major;
    let den = num + pi_minor * l_minor;
    num / den
}

fn post_major_anc_fixed(n: usize, l_a: f64, l_other_mean: f64, usfs: &[f64]) -> f64 {
    let pi0 = usfs[0].max(1e-300);
    let pin = usfs[n].max(1e-300);
    let num = pi0 * l_a;
    let den = num + pin * l_other_mean;
    num / den
}

fn ptree(n_outgroups: usize, focal: i8, out: &[i8], cache: &[f64]) -> Vec<f64> {
    match n_outgroups {
        1 => vec![1.0],
        2 => {
            let o1 = out[0];
            let o2 = out[1];
            let mut v = vec![0.0; 4];
            for b2u in 0..4 {
                let b2 = b2u as i8;
                v[b2u] = p_branch(cache, 0, focal, b2)
                    * p_branch(cache, 1, b2, o1)
                    * p_branch(cache, 2, b2, o2);
            }
            v
        }
        3 => {
            let o1 = out[0];
            let o2 = out[1];
            let o3 = out[2];
            let mut v = vec![0.0; 16];
            for b2u in 0..4 {
                let b2 = b2u as i8;
                for b4u in 0..4 {
                    let b4 = b4u as i8;
                    let idx = 4 * b2u + b4u; // b2 outer, b4 inner
                    v[idx] = p_branch(cache, 0, focal, b2)
                        * p_branch(cache, 1, b2, o1)
                        * p_branch(cache, 2, b2, b4)
                        * p_branch(cache, 3, b4, o2)
                        * p_branch(cache, 4, b4, o3);
                }
            }
            v
        }
        _ => unimplemented!("ptree for >3 outgroups not implemented"),
    }
}

fn normalize(v: &mut [f64]) {
    let s: f64 = v.iter().sum();
    if s > 0.0 {
        for x in v.iter_mut() {
            *x /= s;
        }
    }
}

pub fn posterior_ancestral_per_site(
    header: &vcf::Header,
    record: &impl vcf::variant::Record,
    //end: u32,
    ac: i32,
    af: f32,
    an: i32,
    ref_allel: &String,
    alt_allel: &String,
    out_bases: Vec<Base>,
    stage1: &SubstitutionModelFit,
    stage2: &UsfsFit,
) -> Result<EstPolarization, Box<dyn std::error::Error>> {
    // Constants and Cache
    let n_outgroups = out_bases.len();

    let cache = build_branch_transition_cache(stage1);
    let n = stage2.n_total as usize;

    let n_total = an as u16;
    let n_alt = ac.max(0) as u16;
    let n_ref = n_total.saturating_sub(n_alt);

    let (major, minor, n_major) = if n_ref >= n_alt {
        (
            Base::from_str(&ref_allel),
            Base::from_str(&alt_allel),
            n_ref,
        )
    } else {
        (
            Base::from_str(&alt_allel),
            Base::from_str(&ref_allel),
            n_alt,
        )
    };

    let outgroup_counts = base_to_states(&out_bases);
    let minor_state = if n_major == n_total { -1 } else { minor as i16 };
    let major_state = major as i16;

    // Probability Calculation
    let p_major_anc = if minor_state < 0 {
        let a = major as i8;
        let l_a = p_config(n_outgroups, a as i16, &outgroup_counts, &cache).max(1e-300);
        let mut sum_other = 0.0;
        for b in 0..4i8 {
            if b == a {
                continue;
            }
            sum_other += p_config(n_outgroups, b as i16, &outgroup_counts, &cache).max(1e-300);
        }
        post_major_anc_fixed(n, l_a, (sum_other / 3.0).max(1e-300), &stage2.usfs)
    } else {
        let n_min = (n_total as usize) - (n_major as usize);
        let c = n_min.min(n - n_min);
        let l_m = p_config(n_outgroups, major_state, &outgroup_counts, &cache).max(1e-300);
        let l_n = p_config(n_outgroups, minor_state, &outgroup_counts, &cache).max(1e-300);
        post_major_anc_polymorphic(n, c, l_m, l_n, &stage2.usfs)
    };

    let mut pt = ptree(n_outgroups, major_state as i8, &outgroup_counts, &cache);
    normalize(&mut pt);

    // Major is ancestral, no swap needed
    let modified_record = if p_major_anc >= 0.5 {
        None
    } else {
        // Need to swap
        let ac_swapped = an - ac;
        let af_swapped = 1.0 - af;

        let mut record_buf = vcf::variant::RecordBuf::try_from_variant_record(header, record)?;

        // Swap alleles
        /*        *record_buf.reference_bases_mut() = alt_allel.parse()?;
         *record_buf.alternate_bases_mut() = vec![ref_allel.to_string()].into();
         */
        // Swap alleles
        *record_buf.reference_bases_mut() = alt_allel.to_string().into();
        *record_buf.alternate_bases_mut() = vec![ref_allel.to_string()].into();

        // Update INFO
        let info_mut = record_buf.info_mut();
        info_mut.insert(
            String::from(key::ALLELE_COUNT),
            Some(InfoValue::Integer(ac_swapped)),
        );
        info_mut.insert(
            String::from(key::ALLELE_FREQUENCIES),
            Some(InfoValue::Float(af_swapped)),
        );

        // Update genotypes
        let samples_buf = std::mem::take(record_buf.samples_mut());
        let (keys, mut cols): (vcf::variant::record_buf::samples::Keys, _) = samples_buf.into();

        let gt_idx_opt = keys.as_ref().iter().position(|k| k == format_key::GENOTYPE);

        if let Some(gt_idx) = gt_idx_opt {
            for sample_fields in &mut cols {
                if let Some(Some(SampleValue::Genotype(genotype))) = sample_fields.get_mut(gt_idx) {
                    for allele in genotype.as_mut().iter_mut() {
                        if let Some(idx) = allele.position_mut() {
                            match *idx {
                                0 => *idx = 1,
                                1 => *idx = 0,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        *record_buf.samples_mut() = vcf::variant::record_buf::Samples::new(keys, cols);

        Some(record_buf)
    };

    Ok(EstPolarization {
        modified_record,
        p_major_anc,
        pt,
    })
}

pub struct EstPolarization {
    pub modified_record: Option<vcf::variant::RecordBuf>,
    pub p_major_anc: f64,
    pub pt: Vec<f64>,
}
