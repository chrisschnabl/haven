use std::fs::File;
use std::io::{BufReader};
use reqwest;
use csv;
use anyhow::{Result, Context};
use std::marker::PhantomData;
use arrow::array::{Int64Array, ArrayRef, Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::path::PathBuf;
use std::io::{stdout, Write};

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
    pub id: String,
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

#[derive(Debug, Clone)]
pub struct ClassificationContent {
    pub id: i64,
    pub question: String,
    pub subject: String,
    pub choices: Vec<String>,
    pub answer: String,
    pub answer_index: i64,
}

impl DatasetContent for ToxicityContent {
    fn parse_record(record: &csv::StringRecord) -> Result<Self> {
        Ok(Self {
            id: record.get(0)
                .context("Missing id field")?
                .to_string(),
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

impl DatasetContent for ClassificationContent {
    fn parse_record(record: &csv::StringRecord) -> Result<Self> {
        Ok(Self {
            id: record.get(0)
                .context("Missing id field")?
                .parse()
                .context("Failed to parse id")?,
            question: record.get(1)
                .context("Missing question field")?
                .to_string(),
            subject: record.get(2)
                .context("Missing subject field")?
                .to_string(),
            choices: record.get(3)
                .context("Missing choices field")?
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
            answer: record.get(4)
                .context("Missing answer field")?
                .to_string(),
            answer_index: 0,
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

    fn ensure_file_exists(&self) -> Result<()> {
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
        Ok(())
    }

    fn read_csv(&self, limit: Option<usize>, start_from: usize) -> Result<Vec<DatasetEntry<T>>> {
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

    pub fn load_or_download(&self, limit: Option<usize>, start_from: usize) -> Result<Vec<DatasetEntry<T>>> {
        self.ensure_file_exists()?;
        self.read_csv(limit, start_from)
    }
}

#[derive(Debug)]
pub struct ParquetDatasetLoader {
    file_path: String,
    url: String,
    cache_dir: Option<PathBuf>,
    batch_size: usize,
}

impl ParquetDatasetLoader {
    pub fn new(file_path: &str, url: &str) -> Self {
        Self {
            file_path: file_path.to_string(),
            url: url.to_string(),
            cache_dir: None,
            batch_size: 1024,
        }
    }

    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = Some(cache_dir);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    fn ensure_file_exists(&self) -> Result<()> {
        if !std::path::Path::new(&self.file_path).exists() {
            println!("Dataset not found, downloading...");
            
            // Create parent directory if it doesn't exist
            if let Some(parent) = std::path::Path::new(&self.file_path).parent() {
                std::fs::create_dir_all(parent)
                    .context("Failed to create parent directory")?;
            }

            // Download with progress indicator
            let response = reqwest::blocking::get(&self.url)
                .context("Failed to download dataset")?;
            
            let total_size = response.content_length()
                .unwrap_or(0);
            
            let mut file = File::create(&self.file_path)
                .context("Failed to create file")?;
            
            let mut downloaded: u64 = 0;
            let mut buffer = vec![0; 8192];
            
            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.context("Failed to read chunk")?;
                file.write_all(&chunk)?;
                downloaded += chunk.len() as u64;
                
                if total_size > 0 {
                    let progress = (downloaded as f64 / total_size as f64) * 100.0;
                    print!("\rDownloading: {:.1}%", progress);
                    stdout().flush()?;
                }
            }
            println!("\nDataset downloaded successfully.");
        }
        Ok(())
    }

    pub fn load_classification_data(&self, limit: Option<usize>, start_from: usize) -> Result<Vec<DatasetEntry<ClassificationContent>>> {
        self.ensure_file_exists()?;
        
        let file = File::open(&self.file_path)
            .context("Failed to open dataset file")?;
        
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .context("Failed to create parquet reader")?;
        
        // Configure reader properties
        let props = builder.properties();
        let mut reader = builder
            .with_batch_size(self.batch_size)
            .build()
            .context("Failed to build parquet reader")?;
        
        let mut entries = Vec::new();
        let mut current_idx = 0;

        while let Some(batch_result) = reader.next() {
            let batch = batch_result.context("Failed to read record batch")?;
            
            // Extract arrays with null checks
            let question_array = batch.column_by_name("question")
                .context("Missing question column")?
                .as_any()
                .downcast_ref::<arrow::array::GenericStringArray<i32>>()
                .context("Failed to get question column as StringArray")?;
                
            let subject_array = batch.column_by_name("subject")
                .context("Missing subject column")?
                .as_any()
                .downcast_ref::<arrow::array::GenericStringArray<i32>>()
                .context("Failed to get subject column as StringArray")?;
                
            let choices_array = batch.column_by_name("choices")
                .context("Missing choices column")?;
                
            let answer_array = batch.column_by_name("answer")
                .context("Missing answer column")?
                .as_any()
                .downcast_ref::<Int64Array>()
                .context("Failed to get answer column as Int64Array")?;

            // Process batch rows
            for row_idx in 0..batch.num_rows() {
                if current_idx < start_from {
                    current_idx += 1;
                    continue;
                }

                if let Some(limit) = limit {
                    if entries.len() >= limit {
                        break;
                    }
                }

                // Extract row data with better error handling
                let question = question_array.is_valid(row_idx)
                    .then(|| question_array.value(row_idx).to_string())
                    .context("Question is null")?;

                let subject = subject_array.is_valid(row_idx)
                    .then(|| subject_array.value(row_idx).to_string())
                    .context("Subject is null")?;

                let choices = self.extract_choices(choices_array, row_idx)
                    .context("Failed to extract choices")?;

                let answer_idx = answer_array.is_valid(row_idx)
                    .then(|| answer_array.value(row_idx))
                    .context("Answer is null")?;

                let answer = choices.get(answer_idx as usize)
                    .context("Answer index out of bounds")?
                    .to_string();

                entries.push(DatasetEntry {
                    id: current_idx,
                    content: ClassificationContent {
                        id: current_idx as i64,
                        question,
                        subject,
                        choices,
                        answer,
                        answer_index: answer_idx,
                    },
                });

                current_idx += 1;
            }

            if let Some(limit) = limit {
                if entries.len() >= limit {
                    break;
                }
            }
        }

        if entries.is_empty() {
            anyhow::bail!("No entries found in the dataset");
        }

        println!("Loaded {} entries from parquet dataset", entries.len());
        Ok(entries)
    }

    /// Helper function to extract choices from an array column with improved error handling
    fn extract_choices(&self, choices_array: &ArrayRef, row_idx: usize) -> Result<Vec<String>> {
        if let Some(list_array) = choices_array.as_any().downcast_ref::<arrow::array::ListArray>() {
            if !list_array.is_valid(row_idx) {
                anyhow::bail!("Choices list is null at row {}", row_idx);
            }

            let values = list_array.value(row_idx);
            if let Some(string_array) = values.as_any().downcast_ref::<arrow::array::GenericStringArray<i32>>() {
                let choices: Vec<String> = (0..string_array.len() as usize)
                    .map(|i| {
                        if !string_array.is_valid(i) {
                            anyhow::bail!("Choice string is null at index {}", i);
                        }
                        string_array.value(i).to_string()
                    })
                    .collect();
                Ok(choices)
            } else {
                anyhow::bail!("Choices values are not strings")
            }
        } else {
            anyhow::bail!("Choices column is not a list array")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_parquet_loader() -> Result<()> {
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test.parquet");
        
        // Create a test parquet file
        let schema = Schema::new(vec![
            Field::new("question", DataType::Utf8, false),
            Field::new("subject", DataType::Utf8, false),
            Field::new("choices", DataType::List(Box::new(Field::new("item", DataType::Utf8, false))), false),
            Field::new("answer", DataType::Int64, false),
        ]);

        let loader = ParquetDatasetLoader::new(
            file_path.to_str().unwrap(),
            "http://example.com/test.parquet"
        );

        // Test loading with limit
        let result = loader.load_classification_data(Some(10), 0);
        assert!(result.is_ok());

        Ok(())
    }
}