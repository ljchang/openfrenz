//! A minimal columnar table and its Parquet writer.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, RecordBatch, StringArray,
    UInt8Array, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use crate::Error;

/// One column of values.
#[derive(Debug, Clone)]
pub enum Column {
    /// 64-bit floats (times).
    F64(Vec<f64>),
    /// 64-bit integers.
    I64(Vec<i64>),
    /// Nullable 64-bit integers.
    OptI64(Vec<Option<i64>>),
    /// 32-bit integers (EEG counts).
    I32(Vec<i32>),
    /// 16-bit integers (IMU counts, sequence numbers).
    I16(Vec<i16>),
    /// Nullable 16-bit integers.
    OptI16(Vec<Option<i16>>),
    /// 8-bit integers (sample position in packet).
    I8(Vec<i8>),
    /// Unsigned 32-bit integers (PPG counts).
    U32(Vec<u32>),
    /// Unsigned bytes.
    U8(Vec<u8>),
    /// Booleans (flags).
    Bool(Vec<bool>),
    /// Text.
    Str(Vec<String>),
}

impl Column {
    fn len(&self) -> usize {
        match self {
            Self::F64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::OptI64(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::OptI16(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::Bool(v) => v.len(),
            Self::Str(v) => v.len(),
        }
    }

    fn to_arrow(&self) -> (DataType, bool, ArrayRef) {
        match self {
            Self::F64(v) => (DataType::Float64, false, Arc::new(Float64Array::from(v.clone()))),
            Self::I64(v) => (DataType::Int64, false, Arc::new(Int64Array::from(v.clone()))),
            Self::OptI64(v) => (DataType::Int64, true, Arc::new(Int64Array::from(v.clone()))),
            Self::I32(v) => (DataType::Int32, false, Arc::new(Int32Array::from(v.clone()))),
            Self::I16(v) => (DataType::Int16, false, Arc::new(Int16Array::from(v.clone()))),
            Self::OptI16(v) => (DataType::Int16, true, Arc::new(Int16Array::from(v.clone()))),
            Self::I8(v) => (DataType::Int8, false, Arc::new(Int8Array::from(v.clone()))),
            Self::U32(v) => (DataType::UInt32, false, Arc::new(UInt32Array::from(v.clone()))),
            Self::U8(v) => (DataType::UInt8, false, Arc::new(UInt8Array::from(v.clone()))),
            Self::Bool(v) => (DataType::Boolean, false, Arc::new(BooleanArray::from(v.clone()))),
            Self::Str(v) => (DataType::Utf8, false, Arc::new(StringArray::from(v.clone()))),
        }
    }
}

/// A named table of equal-length columns.
#[derive(Debug, Clone, Default)]
pub struct Table {
    /// Column names and values, in order.
    pub columns: Vec<(String, Column)>,
}

impl Table {
    /// Adds a column.
    pub fn with(mut self, name: &str, column: Column) -> Self {
        self.columns.push((name.to_owned(), column));
        self
    }

    /// Number of rows (0 for a table without columns).
    pub fn rows(&self) -> usize {
        self.columns.first().map_or(0, |(_, c)| c.len())
    }

    /// Writes the table as a zstd-compressed Parquet file.
    pub fn write_parquet(&self, path: &Path) -> Result<(), Error> {
        let rows = self.rows();
        if let Some((name, _)) = self.columns.iter().find(|(_, c)| c.len() != rows) {
            return Err(Error::Invalid(format!("column {name} length differs from {rows}")));
        }
        let mut fields = Vec::new();
        let mut arrays = Vec::new();
        for (name, column) in &self.columns {
            let (dtype, nullable, array) = column.to_arrow();
            fields.push(Field::new(name, dtype, nullable));
            arrays.push(array);
        }
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), arrays)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .build();
        let mut writer = ArrowWriter::try_new(File::create(path)?, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }
}
