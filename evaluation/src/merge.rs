
fn merge_responses() -> Result<()> {
    let mut all_batches = Vec::new();
    
    for entry in std::fs::read_dir(".")? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() && 
           path.to_string_lossy().starts_with("./llama_responses_") && 
           path.to_string_lossy().ends_with(".parquet") {
            
            println!("Reading {}", path.display());
            let file = File::open(path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            let mut reader = builder.build()?;
            
            if let Some(batch) = reader.next() {
                all_batches.push(batch?);
            }
        }
    }
    
    if all_batches.is_empty() {
        anyhow::bail!("No parquet files found to merge");
    }
    
    let schema = all_batches[0].schema();
    let mut combined_ids = Vec::new();
    let mut combined_inputs = Vec::new();
    let mut combined_responses = Vec::new();
    let mut combined_toxics = Vec::new();
    
    for batch in all_batches {
        let id_array = batch.column(0).as_any().downcast_ref::<Int64Array>()
            .context("Failed to get id column")?;
        let input_array = batch.column(1).as_any().downcast_ref::<StringArray>()
            .context("Failed to get input column")?;
        let response_array = batch.column(2).as_any().downcast_ref::<StringArray>()
            .context("Failed to get response column")?;
        let toxic_array = batch.column(3).as_any().downcast_ref::<Float64Array>()
            .context("Failed to get toxic column")?;
        
        for i in 0..batch.num_rows() {
            combined_ids.push(id_array.value(i));
            combined_inputs.push(input_array.value(i).to_string());
            combined_responses.push(response_array.value(i).to_string());
            combined_toxics.push(toxic_array.value(i));
        }
    }
    
    let final_batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(combined_ids)),
            Arc::new(StringArray::from(combined_inputs)),
            Arc::new(StringArray::from(combined_responses)),
            Arc::new(Float64Array::from(combined_toxics)),
        ],
    )?;
    
    let output_file = File::create("llama_responses.parquet")?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(output_file, final_batch.schema(), Some(props))?;
    
    writer.write(&final_batch)?;
    writer.close()?;
    
    println!("Successfully merged {} responses into llama_responses.parquet", final_batch.num_rows());
    
    for entry in std::fs::read_dir(".")? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() && 
           path.to_string_lossy().starts_with("./llama_responses_") && 
           path.to_string_lossy().ends_with(".parquet") {
            println!("Could remove {}", path.display());
            //std::fs::remove_file(path)?;
        }
    }
    
    Ok(())
}