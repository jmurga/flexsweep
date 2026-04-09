use flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;
use std::str::FromStr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MafError {
    #[error("Invalid MAF format: {0}")]
    ParseError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

#[derive(Debug, Clone)]
pub struct MafHeader {
    #[allow(dead_code)]
    pub version: String,
    #[allow(dead_code)]
    pub scoring: Option<String>,
    #[allow(dead_code)]
    pub program: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub src: String,
    pub start: u64,
    pub size: u64,
    pub strand: Strand,
    #[allow(dead_code)]
    pub src_size: u64,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Strand {
    Forward,
    Reverse,
}

impl FromStr for Strand {
    type Err = MafError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "+" => Ok(Strand::Forward),
            "-" => Ok(Strand::Reverse),
            _ => Err(MafError::ParseError("Invalid strand".to_string())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlignmentBlock {
    #[allow(dead_code)]
    pub score: Option<f64>,
    #[allow(dead_code)]
    pub pass: Option<u32>,
    pub sequences: Vec<Sequence>,
}

fn is_gzipped(filepath: &Path) -> io::Result<bool> {
    let mut file = File::open(filepath)?;
    let mut magic_bytes = [0_u8; 2];
    file.read_exact(&mut magic_bytes)?;
    Ok(magic_bytes == [0x1f, 0x8b])
}

fn get_reader(filepath: &Path) -> io::Result<Box<dyn BufRead>> {
    let file = File::open(filepath)?;

    if is_gzipped(filepath)? {
        Ok(Box::new(BufReader::new(MultiGzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

type BoxedReader = Box<dyn BufRead>;

pub struct MafReader {
    reader: BoxedReader,
    header: Option<MafHeader>,
    current_line: String,
}

impl MafReader {
    pub fn from_file(filepath: &Path) -> io::Result<Self> {
        let reader = get_reader(filepath)?;
        Ok(Self {
            reader,
            header: None,
            current_line: String::new(),
        })
    }

    pub fn read_header(&mut self) -> Result<&MafHeader, MafError> {
        if self.header.is_some() {
            return Ok(self.header.as_ref().unwrap());
        }

        self.current_line.clear();
        self.reader.read_line(&mut self.current_line)?;

        if !self.current_line.starts_with("##maf") {
            return Err(MafError::ParseError("Missing ##maf header".to_string()));
        }

        let mut version = None;
        let mut scoring = None;
        let mut program = None;

        for pair in self.current_line["##maf".len()..].split_whitespace() {
            let mut parts = pair.split('=');
            match (parts.next(), parts.next()) {
                (Some("version"), Some(v)) => version = Some(v.to_string()),
                (Some("scoring"), Some(s)) => scoring = Some(s.to_string()),
                (Some("program"), Some(p)) => program = Some(p.to_string()),
                _ => continue,
            }
        }

        self.header = Some(MafHeader {
            version: version.ok_or_else(|| MafError::ParseError("Missing version".to_string()))?,
            scoring,
            program,
        });

        Ok(self.header.as_ref().unwrap())
    }

    pub fn next_block(&mut self) -> Result<Option<AlignmentBlock>, MafError> {
        let mut block: Option<AlignmentBlock> = None;
        let mut sequences: Vec<Sequence> = Vec::new();

        self.current_line.clear();
        while self.reader.read_line(&mut self.current_line)? > 0 {
            let line = self.current_line.trim();
            if line.is_empty() {
                if block.is_some() {
                    let mut b = block.take().unwrap();
                    b.sequences = sequences;
                    return Ok(Some(b));
                }
                continue;
            }

            match &line[0..1] {
                "a" => {
                    let mut score = None;
                    let mut pass = None;

                    for pair in line[1..].split_whitespace() {
                        let mut parts = pair.split('=');
                        match (parts.next(), parts.next()) {
                            (Some("score"), Some(s)) => {
                                score = Some(s.parse().map_err(|_| {
                                    MafError::ParseError("Invalid score".to_string())
                                })?)
                            }
                            (Some("pass"), Some(p)) => {
                                pass = Some(p.parse().map_err(|_| {
                                    MafError::ParseError("Invalid pass".to_string())
                                })?)
                            }
                            _ => continue,
                        }
                    }

                    block = Some(AlignmentBlock {
                        score,
                        pass,
                        sequences: Vec::new(),
                    });
                }
                "s" => {
                    let parts: Vec<&str> = line[1..].split_whitespace().collect();
                    if parts.len() != 6 {
                        return Err(MafError::ParseError("Invalid sequence line".to_string()));
                    }

                    sequences.push(Sequence {
                        src: parts[0].to_string(),
                        start: parts[1].parse().map_err(|_| {
                            MafError::ParseError("Invalid start position".to_string())
                        })?,
                        size: parts[2]
                            .parse()
                            .map_err(|_| MafError::ParseError("Invalid size".to_string()))?,
                        strand: parts[3].parse()?,
                        src_size: parts[4]
                            .parse()
                            .map_err(|_| MafError::ParseError("Invalid source size".to_string()))?,
                        text: parts[5].to_string(),
                    });
                }
                _ => (),
            }
            self.current_line.clear();
        }

        if block.is_some() {
            let mut b = block.take().unwrap();
            b.sequences = sequences;
            Ok(Some(b))
        } else {
            Ok(None)
        }
    }
}
