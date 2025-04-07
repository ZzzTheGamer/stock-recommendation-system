@echo off
echo 启动股票推荐系统后端服务 (GPU模式)...

:: 设置CUDA环境变量
set CUDA_VISIBLE_DEVICES=0

:: 显示环境信息
echo.
echo ===== GPU环境信息 =====
python -c "import torch; print('CUDA是否可用:', torch.cuda.is_available()); print('可用GPU数量:', torch.cuda.device_count()); print('GPU型号:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

:: 如果CUDA不可用，提示错误信息并询问是否继续
python -c "import torch; print(torch.cuda.is_available())" | findstr "False" > nul
if %errorlevel% equ 0 (
    echo.
    echo 警告: CUDA不可用! 可能是以下原因:
    echo   1. 您的电脑没有NVIDIA GPU
    echo   2. 没有安装CUDA工具包
    echo   3. 没有安装支持CUDA的PyTorch版本
    echo   4. GPU驱动程序版本不兼容
    echo.
    set /p CONTINUE="是否继续使用CPU模式运行? (Y/N): "
    if /i not "%CONTINUE%" == "Y" exit
)

:: 启动Flask后端
echo.
echo 启动Flask后端服务...
python app.py

echo.
echo 后端服务已停止。
pause 