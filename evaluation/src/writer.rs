use std::fs::File;
use std::sync::Arc;
use anyhow::{Result, Context};
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use arrow::datatypes::Schema;
use parquet::arrow::arrow_writer::ArrowWriter;

use crate::config::OutputConfig;

pub struct ParquetWriter {
    schema: Arc<Schema>,
    writer: ArrowWriter<File>,
}

// TODO: make this batchign at some point
impl ParquetWriter {
    pub fn new(schema: Schema, config: OutputConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.output_dir)
            .context("Failed to create output directory")?;
        
        let file_path = config.output_dir.join(format!("{}.parquet", config.file_prefix));
            let file = File::create(&file_path)
            .context(format!("Failed to create file: {:?}", file_path))?;
        
        let writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), None)
            .context("Failed to create Arrow writer")?;
        
        Ok(Self {
            schema: Arc::new(schema),
            writer,
        })
    }

    pub fn add_row(&mut self, _idx: usize, row_data: Vec<Arc<dyn Array>>) -> Result<()> {
        let batch = RecordBatch::try_new(
            self.schema.clone(),
            row_data,
        )?;
        
        self.writer.write(&batch)?;

        self.writer.flush()?;
        
        Ok(())
    }
    
    pub fn close(self) -> Result<()> {
        self.writer.close().context("Failed to close writer")?;
        Ok(())
    }
} 