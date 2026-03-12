use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyConnectionError};
use pyo3::types::PyBytes;

use hmac::{Hmac, Mac};
use sha2::Sha256;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// Définition de notre type HMAC
type HmacSha256 = Hmac<Sha256>;

/// Hiérarchie des Tiers de Transport VRAMancer
#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub enum TransportTier {
    DirectRdma,     // Niveau 1: GPUDirect / InfiniBand (Bypass CPU Total) - Futur
    ZeroCopyTcp,    // Niveau 2: Safetensors Buffer partagé + TCP natif
    StandardTcp,    // Niveau 3: Fallback Python Pickle + TCP local
}

/// Détecte le meilleur tier réseau disponible sur ce nœud host
#[pyfunction]
fn detect_best_transport() -> TransportTier {
    // Plus tard, nous ajouterons ici la détection de "nvidia_peermem" / "ibverbs".
    // Pour l'instant, on active notre nouveau "Niveau 2" par défaut (ZeroCopy TCP) !
    TransportTier::ZeroCopyTcp
}

/// (Option B - Le Data Plane Tokio Zéro-Copie)
/// Accepte directement des MemoryViews ou Safetensors bytes sans passer par Pickle
#[pyfunction]
fn send_tensor_p2p(
    py: Python, 
    host: String, 
    port: u16, 
    secret: &[u8], 
    payload: &[u8]
) -> PyResult<Py<PyBytes>> {
    
    // 1. Signature ultra-rapide (C-speed)
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|_| PyValueError::new_err("Erreur Secret HMAC"))?;
    mac.update(payload);
    let signature = mac.finalize().into_bytes();

    let payload_len = payload.len() as u64;
    // La longueur totale comprendra les 32 octets de la signature HMAC
    let total_len = payload_len + 32;

    // 2. Relâche du GIL Python et passage en I/O asynchrone natif
    let result: Result<Vec<u8>, String> = py.allow_threads(|| {
        // Lancement d'un runtime Tokio temporaire dédié au transfert massif
        let rt = tokio::runtime::Runtime::new().unwrap();
        
        rt.block_on(async {
            let addr = format!("{}:{}", host, port);
            
            // Connexion asynchrone
            let mut stream = TcpStream::connect(&addr).await
                .map_err(|e| format!("Echec connexion TCP vers {}: {}", addr, e))?;

            // Envoi optimisé du Header (Total Len)
            stream.write_u64(total_len).await
                .map_err(|e| format!("Echec envoi Header: {}", e))?;
            
            // Envoi de la Signature Zero-Trust
            stream.write_all(&signature).await
                .map_err(|e| format!("Echec envoi Signature: {}", e))?;
                
            // Envoi du Tenseur de plusieurs Go
            stream.write_all(payload).await
                .map_err(|e| format!("Echec envoi Payload: {}", e))?;

            // Attente (sans bloquer Python) de la longueur de la réponse
            let resp_len = stream.read_u64().await
                .map_err(|e| format!("Echec lecture réponse Header: {}", e))?;

            // Allocation et lecture de la réponse
            let mut resp_data = vec![0u8; resp_len as usize];
            stream.read_exact(&mut resp_data).await
                .map_err(|e| format!("Echec lecture réponse Body: {}", e))?;

            Ok(resp_data)
        })
    });

    // 3. Retour en Python : Transformation des bytes C en PyBytes
    match result {
        Ok(data) => Ok(PyBytes::new(py, &data).into()),
        Err(e) => Err(PyConnectionError::new_err(e)),
    }
}

/// Une fonction très rapide pour signer un payload en Rust et esquiver le GIL de Python.
/// Utilisée pour la validation locale.
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
    m.add_class::<TransportTier>()?;
    m.add_function(wrap_pyfunction!(detect_best_transport, m)?)?;
    m.add_function(wrap_pyfunction!(sign_payload_fast, m)?)?;
    m.add_function(wrap_pyfunction!(verify_hmac_fast, m)?)?;
    m.add_function(wrap_pyfunction!(send_tensor_p2p, m)?)?;
    Ok(())
}
