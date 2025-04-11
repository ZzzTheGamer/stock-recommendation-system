@echo off
chcp 65001
echo ===== Start the back-end service (CPU mode) =====
echo.

:: Display environment information
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Number of available GPUs:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('GPU model:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

:: If CUDA is not available, display a warning
python -c "import torch; cuda=torch.cuda.is_available(); print('Warning: CUDA is not available, the service will run in CPU mode, and the performance may be lower' if not cuda else '')"

echo.
echo Note: Running in CPU mode, the calculation speed may be slower
echo If you want to use GPU acceleration, please run start_gpu.bat

echo.
echo Start the Flask application...
python app.py

echo.
echo The back-end service has been stopped.
pause 