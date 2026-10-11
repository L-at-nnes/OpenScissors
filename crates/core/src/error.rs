use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Msg(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("model runtime error: {0}")]
    Ort(String),
    #[error("download failed: {0}")]
    Download(String),
    #[error("checksum mismatch for model '{0}'")]
    Checksum(String),
    #[error("cancelled")]
    Cancelled,
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl<R> From<ort::Error<R>> for Error {
    fn from(e: ort::Error<R>) -> Self {
        Error::Ort(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn msg(s: impl Into<String>) -> Self {
        Error::Msg(s.into())
    }
}
