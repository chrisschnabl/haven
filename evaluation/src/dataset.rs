use std::fs::File;
use std::io::{BufReader};
use reqwest;
use csv;
use anyhow::{Result, Context};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct DatasetEntry<T> {
    pub id: usize,
    pub content: T,
}

pub trait DatasetContent: Sized {
    fn parse_record(record: &csv::StringRecord) -> Result<Self>;
}

#[derive(Debug, Clone)]
pub struct ToxicityContent {
    pub id: i64,
    pub input: String,
    pub response: String,
    pub toxic: f64,
}

#[derive(Debug, Clone)]
pub struct SimilarityContent {
    pub id: i64,
    pub dialogue: String,
    pub summary: String,
}

impl DatasetContent for ToxicityContent {
    fn parse_record(record: &csv::StringRecord) -> Result<Self> {
        Ok(Self {
            id: record.get(0)
                .context("Missing id field")?
                .parse()
                .context("Failed to parse id")?,
            input: record.get(1)
                .context("Missing input field")?
                .to_string(),
            response: record.get(2)
                .context("Missing response field")?
                .to_string(),
            toxic: record.get(4)
                .context("Missing toxic field")?
                .parse()
                .context("Failed to parse toxicity")?,
        })
    }
}

impl DatasetContent for SimilarityContent {
    fn parse_record(record: &csv::StringRecord) -> Result<Self> {
        Ok(Self {
            id: record.get(0)
                .context("Missing id field")?
                .parse()
                .context("Failed to parse id")?,
            dialogue: record.get(1)
                .context("Missing dialogue field")?
                .to_string(),
            summary: record.get(2)
                .context("Missing summary field")?
                .to_string(),
        })
    }
}

#[derive(Debug)]
pub struct DatasetLoader<T: DatasetContent> {
    file_path: String,
    url: String,
    _phantom: PhantomData<T>,
}

impl<T: DatasetContent> DatasetLoader<T> {
    pub fn new(file_path: &str, url: &str) -> Self {
        Self {
            file_path: file_path.to_string(),
            url: url.to_string(),
            _phantom: PhantomData,
        }
    }

    pub fn load_or_download(&self, limit: Option<usize>, start_from: usize) -> Result<Vec<DatasetEntry<T>>> {
        if !std::path::Path::new(&self.file_path).exists() {
            println!("Dataset not found, downloading...");
            let response = reqwest::blocking::get(&self.url)
                .context("Failed to download dataset")?;
            let mut file = File::create(&self.file_path)
                .context("Failed to create file")?;
            std::io::copy(&mut response.bytes()?.as_ref(), &mut file)
                .context("Failed to write dataset to file")?;
            println!("Dataset downloaded successfully.");
        }
        
        let file = File::open(&self.file_path)
            .context("Failed to open dataset file")?;
        let reader = BufReader::new(file);
        let mut csv_reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(reader);
        
        let mut entries = Vec::new();
        for (idx, result) in csv_reader.records().enumerate() {
            if idx < start_from {
                continue;
            }
            
            if let Some(limit) = limit {
                if entries.len() >= limit {
                    break;
                }
            }
            
            let record = result.context("Failed to read CSV record")?;
            entries.push(DatasetEntry {
                id: idx,
                content: T::parse_record(&record)?,
            });
        }
        
        if entries.is_empty() {
            anyhow::bail!("No entries found in the dataset");
        }
        
        println!("Loaded {} entries from dataset", entries.len());
        
        Ok(entries)
    }
} 