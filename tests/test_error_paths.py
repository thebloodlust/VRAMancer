import pytest
from unittest.mock import MagicMock, patch
from core.inference_pipeline import InferencePipeline

def test_pipeline_shutdown_survives_exceptions():
    # Arrange
    pipeline = InferencePipeline()
    
    # Mocking components with MagicMocks that raise exceptions on stop methods
    mock_fault_manager = MagicMock()
    mock_fault_manager.stop.side_effect = RuntimeError("fault_manager error")
    pipeline.fault_manager = mock_fault_manager
    
    mock_continuous_batcher = MagicMock()
    mock_continuous_batcher.stop.side_effect = RuntimeError("batcher error")
    pipeline.continuous_batcher = mock_continuous_batcher
    
    pipeline.stop_rebalancing = MagicMock()
    
    mock_monitor = MagicMock()
    mock_monitor.stop_polling.side_effect = RuntimeError("monitor error")
    pipeline.monitor = mock_monitor
    
    mock_stream_manager = MagicMock()
    mock_stream_manager.stop_monitoring.side_effect = RuntimeError("stream_manager error")
    pipeline.stream_manager = mock_stream_manager
    
    mock_gpu_hotplug = MagicMock()
    mock_gpu_hotplug.stop.side_effect = RuntimeError("hotplug error")
    pipeline.gpu_hotplug = mock_gpu_hotplug
    
    mock_discovery = MagicMock()
    mock_discovery.stop.side_effect = RuntimeError("discovery error")
    pipeline.discovery = mock_discovery
    
    mock_transfer_manager = MagicMock()
    mock_transfer_manager.shutdown.side_effect = RuntimeError("transfer_manager error")
    pipeline.transfer_manager = mock_transfer_manager
    
    # Act
    try:
        pipeline.shutdown()
        success = True
    except Exception as e:
        success = False
        pytest.fail(f"Pipeline shutdown raised an exception instead of handling it gently: {e}")
        
    # Assert
    assert success is True
    # Verify that all the stop methods were actually called despite previous exceptions
    mock_fault_manager.stop.assert_called_once()
    mock_continuous_batcher.stop.assert_called_once()
    mock_monitor.stop_polling.assert_called_once()
    mock_stream_manager.stop_monitoring.assert_called_once()
    mock_gpu_hotplug.stop.assert_called_once()
    mock_discovery.stop.assert_called_once()
    mock_transfer_manager.shutdown.assert_called_once()

@patch('core.inference_pipeline.metrics_server_start', create=True)
def test_pipeline_init_survives_metrics_exception(mock_metrics):
    mock_metrics.side_effect = Exception("Prometheus address already in use")
    
    try:
        pipeline = InferencePipeline()
        success = True
    except Exception as e:
        success = False
        if "Prometheus" in str(e):
            pytest.fail(f"Pipeline init failed due to metrics exception: {e}")
        
    assert success is True

@patch('core.inference_pipeline.get_woi_manager', create=True)
def test_pipeline_generate_survives_wake_exception(mock_get_manager):
    # Setup wake failure
    mock_manager = MagicMock()
    mock_manager.wake_all.side_effect = Exception("WakeOnLan not permitted by OS")
    mock_get_manager.return_value = mock_manager

    pipeline = InferencePipeline()
    
    try:
        # We don't actually want to generate stuff fully
        result = list(pipeline.generate("Test prompt", max_tokens=1))
        # Might fail due to no backend, but shouldn't fail due to WakeOnLan
    except Exception as e:
        if "WakeOnLan" in str(e):
            pytest.fail(f"Pipeline generate failed due to Wake exception: {e}")
            
    assert True