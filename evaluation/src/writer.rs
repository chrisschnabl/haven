use std::fs::File;
use std::sync::Arc;
use anyhow::{Result, Context};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use arrow::datatypes::Schema;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::config::OutputConfig;

const DEFAULT_BATCH_SIZE: usize = 1024;
const DEFAULT_COMPRESSION: Compression = Compression::SNAPPY;

pub struct ParquetWriter {
    schema: Arc<Schema>,
    writer: ArrowWriter<File>,
    batch_size: usize,
    current_batch: Vec<Vec<Arc<dyn Array>>>,
    compression: Compression,
}

impl ParquetWriter {
    pub fn new(schema: Schema, config: OutputConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.output_dir)
            .context("Failed to create output directory")?;
        
        let file_path = config.output_dir.join(format!("{}_{}.parquet", config.file_prefix, 1));
        let file = File::create(&file_path)
            .context(format!("Failed to create file: {:?}", file_path))?;
        
        // Configure writer properties with compression
        let props = WriterProperties::builder()
            .set_compression(DEFAULT_COMPRESSION)
            .build();
        
        let writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props))
            .context("Failed to create Arrow writer")?;
        
        Ok(Self {
            schema: Arc::new(schema),
            writer,
            batch_size: DEFAULT_BATCH_SIZE,
            current_batch: Vec::new(),
            compression: DEFAULT_COMPRESSION,
        })
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    pub fn add_row(&mut self, _idx: usize, row_data: Vec<Arc<dyn Array>>) -> Result<()> {
        // Initialize current_batch if empty
        if self.current_batch.is_empty() {
            self.current_batch = vec![Vec::with_capacity(self.batch_size); row_data.len()];
        }

        // Add row data to current batch
        for (i, array) in row_data.into_iter().enumerate() {
            self.current_batch[i].push(array);
        }

        // Write batch if it's full
        if self.current_batch[0].len() >= self.batch_size {
            self.write_batch()?;
        }
        
        Ok(())
    }

    fn write_batch(&mut self) -> Result<()> {
        if self.current_batch.is_empty() {
            return Ok(());
        }

        // Convert vectors of arrays into record batch
        let arrays: Vec<Arc<dyn Array>> = self.current_batch
            .iter_mut()
            .map(|arrays| {
                let array = arrays.remove(0);
                arrays.shrink_to_fit();
                array
            })
            .collect();

        let batch = RecordBatch::try_new(
            self.schema.clone(),
            arrays,
        )?;
        
        self.writer.write(&batch)?;
        self.writer.flush()?;
        
        Ok(())
    }
    
    pub fn close(mut self) -> Result<()> {
        // Write any remaining rows
        self.write_batch()?;
        
        // Close the writer
        self.writer.close().context("Failed to close writer")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use std::path::Path;

    #[test]
    fn test_parquet_writer() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Utf8, false),
        ]);

        let config = OutputConfig {
            output_dir: temp_dir.path().to_path_buf(),
            file_prefix: "test".to_string(),
        };

        let mut writer = ParquetWriter::new(schema, config)?;

        // Write some test data
        for i in 0..5 {
            writer.add_row(i, vec![
                Arc::new(Int64Array::from(vec![i as i64])),
                Arc::new(StringArray::from(vec![format!("value_{}", i)])),
            ])?;
        }

        writer.close()?;

        // Verify file was created
        let file_path = temp_dir.path().join("test_1.parquet");
        assert!(file_path.exists());

        Ok(())
    }
} 