@echo off
chcp 65001
echo ===== 启动后端服务 (CPU模式) =====
echo.

:: 显示环境信息
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA是否可用:', torch.cuda.is_available()); print('可用GPU数量:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('GPU型号:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

:: 如果CUDA不可用，显示警告
python -c "import torch; cuda=torch.cuda.is_available(); print('警告: CUDA不可用，将使用CPU模式运行，性能可能较低' if not cuda else '')"

echo.
echo 注意: 正在使用CPU模式运行，计算速度可能较慢
echo 如果您想使用GPU加速，请运行 start_gpu.bat

echo.
echo 启动Flask应用...
python app.py

echo.
echo 后端服务已停止。
pause 