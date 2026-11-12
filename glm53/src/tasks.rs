//! A deterministic synthetic Python corpus.
//!
//! Mirrors the shape of `glm53_flash/tasks.py`: eight operation families, each
//! with a prompt and a reference completion. The generator is deterministic
//! but uses its own PRNG, so the exact bytes differ from the Python reference.

/// One operation family.
#[derive(Clone, Copy, Debug)]
pub struct Family {
    pub name: &'static str,
    pub arguments: &'static str,
    pub descriptions: &'static [&'static str],
    pub body: &'static str,
}

pub const FAMILIES: &[Family] = &[
    Family {
        name: "increment",
        arguments: "x",
        descriptions: &["Return x plus one.", "Increase x by one."],
        body: "\n    return x + 1\n",
    },
    Family {
        name: "double",
        arguments: "x",
        descriptions: &["Return two times x.", "Double the input."],
        body: "\n    return x * 2\n",
    },
    Family {
        name: "square",
        arguments: "x",
        descriptions: &["Return x squared.", "Multiply x by itself."],
        body: "\n    return x * x\n",
    },
    Family {
        name: "absolute",
        arguments: "x",
        descriptions: &[
            "Return the absolute value of x.",
            "Make a negative x positive.",
        ],
        body: "\n    return -x if x < 0 else x\n",
    },
    Family {
        name: "nonnegative",
        arguments: "x",
        descriptions: &[
            "Clamp x to at least zero.",
            "Return zero when x is negative.",
        ],
        body: "\n    return x if x > 0 else 0\n",
    },
    Family {
        name: "even",
        arguments: "x",
        descriptions: &["Return whether x is even.", "Check divisibility by two."],
        body: "\n    return x % 2 == 0\n",
    },
    Family {
        name: "reverse",
        arguments: "text",
        descriptions: &["Return text in reverse order.", "Reverse the string."],
        body: "\n    return text[::-1]\n",
    },
    Family {
        name: "list_sum",
        arguments: "values",
        descriptions: &["Return the sum of values.", "Add every number in the list."],
        body: "\n    return sum(values)\n",
    },
];

/// A splitmix64 generator: small, deterministic and good enough for a corpus.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `[0, bound)`.
    pub fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }

    /// A value in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        // 53 bits of mantissa, so the value is uniform on the representable grid.
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// A standard normal draw (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// A random lower-case letter.
    pub fn letter(&mut self) -> char {
        (b'a' + self.below(26) as u8) as char
    }
}

/// A generated coding task.
#[derive(Clone, Debug)]
pub struct CodingTask {
    pub task_id: String,
    pub family: String,
    pub entry_point: String,
    pub prompt: String,
    pub reference_completion: String,
}

impl CodingTask {
    /// The full reference source, prompt plus completion.
    pub fn reference_source(&self) -> String {
        format!("{}{}", self.prompt, self.reference_completion)
    }
}

/// Build one task for a family.
pub fn make_task(family: &Family, split: &str, index: usize, seed: u64) -> CodingTask {
    let name_sum: u64 = family.name.bytes().map(u64::from).sum();
    let mut rng = Rng::new(
        seed.wrapping_add(1)
            .wrapping_mul(1_000_003)
            .wrapping_add((index as u64).wrapping_mul(9176))
            .wrapping_add(name_sum),
    );

    let suffix: String = (0..7).map(|_| rng.letter()).collect();
    let entry_point = format!("{}_{}", family.name, suffix);
    let description = family.descriptions[rng.below(family.descriptions.len())];

    let prompt = format!(
        "# Complete this Python function.\n# {description}\ndef {entry_point}({}):",
        family.arguments
    );

    CodingTask {
        task_id: format!("{split}-{}-{index:03}", family.name),
        family: family.name.to_string(),
        entry_point,
        prompt,
        reference_completion: family.body.to_string(),
    }
}

/// One document of the pretraining corpus.
pub fn pretraining_text(index: usize, seed: u64) -> String {
    let mut rng = Rng::new(seed.wrapping_mul(10_000_019).wrapping_add(index as u64));
    let family = &FAMILIES[rng.below(FAMILIES.len())];
    let task = make_task(family, "pretrain", index, seed);
    task.reference_source()
}

/// A frozen evaluation split: `per_family` tasks for every family.
pub fn frozen_tasks(split: &str, per_family: usize) -> Vec<CodingTask> {
    let seeds: &[(&str, u64)] = &[
        ("dev", 1701),
        ("final", 2909),
        ("rl", 3907),
        ("confirm", 8123),
    ];

    let base = seeds
        .iter()
        .find(|(name, _)| *name == split)
        .map(|(_, seed)| *seed)
        .unwrap_or_else(|| panic!("unknown split: {split}"));

    let mut tasks = Vec::new();
    for (family_index, family) in FAMILIES.iter().enumerate() {
        let seed = base + family_index as u64 * 53;
        for index in 0..per_family {
            tasks.push(make_task(family, split, index, seed));
        }
    }
    tasks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretraining_text_is_deterministic() {
        assert_eq!(pretraining_text(7, 42), pretraining_text(7, 42));
    }

    #[test]
    fn pretraining_text_has_a_prompt_and_a_body() {
        let text = pretraining_text(3, 42);
        assert!(text.starts_with("# Complete this Python function.\n# "));
        assert!(text.contains("\n    return "));
    }

    /// The 64-task confirm split is a deterministic superset of the 32-task one,
    /// so raising the power does not change the tasks already reported on.
    #[test]
    fn the_confirm_split_extends_deterministically() {
        let small = frozen_tasks("confirm", 4);
        let large = frozen_tasks("confirm", 8);
        assert_eq!(small.len(), 32);
        assert_eq!(large.len(), 64);
        for task in &small {
            assert!(
                large.iter().any(|other| other.task_id == task.task_id),
                "{} missing from the 64-task split",
                task.task_id
            );
        }
        let ids = |tasks: Vec<CodingTask>| -> Vec<String> {
            tasks.into_iter().map(|task| task.task_id).collect()
        };
        assert_eq!(ids(large), ids(frozen_tasks("confirm", 8)));
    }

    #[test]
    fn frozen_split_covers_every_family() {
        let tasks = frozen_tasks("dev", 4);
        assert_eq!(tasks.len(), FAMILIES.len() * 4);
        for family in FAMILIES {
            assert!(tasks.iter().any(|task| task.family == family.name));
        }
    }

    #[test]
    fn frozen_splits_differ() {
        let dev = frozen_tasks("dev", 2);
        let confirm = frozen_tasks("confirm", 2);
        assert_ne!(dev[0].entry_point, confirm[0].entry_point);
    }
}
