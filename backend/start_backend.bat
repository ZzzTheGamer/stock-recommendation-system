@echo off
echo Start the stock recommendation system back-end service (GPU mode)...

:: Set CUDA environment variable
set CUDA_VISIBLE_DEVICES=0

:: Display environment information
echo.
echo ===== GPU environment information =====
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Number of available GPUs:', torch.cuda.device_count()); print('GPU model:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

:: If CUDA is not available, prompt an error message and ask if you want to continue
python -c "import torch; print(torch.cuda.is_available())" | findstr "False" > nul
if %errorlevel% equ 0 (
    echo.
    echo Warning: CUDA is not available! Possible reasons:
    echo   1. Your computer does not have a NVIDIA GPU
    echo   2. CUDA tool package is not installed
    echo   3. PyTorch version without CUDA support is installed
    echo   4. GPU driver version is not compatible
    echo.
    set /p CONTINUE="Do you want to continue using the CPU mode? (Y/N): "
    if /i not "%CONTINUE%" == "Y" exit
)

:: Start Flask back-end
echo.
echo Start the Flask back-end service...
python app.py

echo.
echo The back-end service has been stopped.
pause 