use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyBytes;

use hmac::{Hmac, Mac};
use sha2::Sha256;

// Définition de notre type HMAC
type HmacSha256 = Hmac<Sha256>;

/// (Option B - Préparation)
/// Fonction qui démontre l'intégration de Tokio pour des tâches I/O asynchrones intenses. 
/// Actuellement, elle s'occupera juste de temporiser sans bloquer le thread OS.
#[pyfunction]
fn rust_sleep_demo(py: Python, ms: u64) -> PyResult<()> {
    // Cette fonction libère le GIL Python (Python peut faire autre chose)
    // pendant que le code C/Rust travaille (ici, il dort, plus tard il fera de la Data-Transfer TCP).
    py.allow_threads(|| {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    });
    Ok(())
}

/// Une fonction très rapide pour signer un payload en Rust et esquiver le GIL de Python.
/// Utilisée pour le réseau P2P Zero-Trust de VRAMancer.
#[pyfunction]
fn sign_payload_fast(py: Python, secret: &[u8], payload: &[u8]) -> PyResult<Py<PyBytes>> {
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|_| PyValueError::new_err("Erreur lors de l'initialisation du Secret HMAC"))?;
    
    mac.update(payload);
    let result = mac.finalize().into_bytes();
    
    // On retourne les bytes directement dans le format natif de Python
    Ok(PyBytes::new(py, &result).into())
}

/// Vérifie ultra-rapidement une signature HMAC-SHA256 en Rust.
#[pyfunction]
fn verify_hmac_fast(_py: Python, secret: &[u8], payload: &[u8], signature: &[u8]) -> PyResult<bool> {
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|_| PyValueError::new_err("Erreur lors de l'initialisation du Secret HMAC"))?;
    
    mac.update(payload);
    
    // Si verify_slice ne panic pas (Ok), la signature est bonne.
    match mac.verify_slice(signature) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// C'est ici que l'on déclare officiellement notre module Python.
#[pymodule]
fn vramancer_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sign_payload_fast, m)?)?;
    m.add_function(wrap_pyfunction!(verify_hmac_fast, m)?)?;
    m.add_function(wrap_pyfunction!(rust_sleep_demo, m)?)?;
    Ok(())
}
